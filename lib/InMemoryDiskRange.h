#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "ConfigReader.h"
#include "Gorder.h"
#include "Kmeans.h"

struct InMemoryClusterData
{
  size_t n = 0;
  size_t d = 0;
  size_t cluster_num = 0;
  size_t max_points = 0;
  std::vector<float> cluster_data;
  std::vector<size_t> bucket_sizes;
  std::vector<size_t> file_pos;
  std::vector<float> centroids;
  std::vector<float> radii;
  std::vector<std::vector<size_t>> assignment;
};

struct InMemoryBuildArtifacts
{
  InMemoryClusterData cluster_data;
  std::unique_ptr<hnswlib::L2Space> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;
};

inline std::vector<float> load_all_data(const std::string& data_file, size_t& n,
                                        size_t& d)
{
  DataReader data_reader(data_file);
  n = data_reader.n;
  d = data_reader.d;

  std::vector<float> raw_data(n * d);
  size_t batch_size = std::max<size_t>(1, n / 1000);
  data_reader.reset();
  for (size_t i = 0; i < div_round_up(n, batch_size); i++)
  {
    size_t num = batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
    data_reader.get_batch((char*)(raw_data.data() + i * batch_size * d), num);
  }

  return raw_data;
}

inline std::vector<float> sample_centroids_from_memory(
    const std::vector<float>& raw_data, size_t n, size_t d, size_t cluster_num)
{
  std::vector<float> centroids(cluster_num * d);
  std::vector<size_t> sample_id(n);
  for (size_t i = 0; i < n; i++) sample_id[i] = i;
  std::srand(283);
  std::random_shuffle(sample_id.begin(), sample_id.end());
  std::sort(sample_id.begin(), sample_id.begin() + cluster_num);
  for (size_t i = 0; i < cluster_num; i++)
  {
    memcpy(centroids.data() + i * d, raw_data.data() + sample_id[i] * d,
           d * sizeof(float));
  }
  return centroids;
}

inline InMemoryClusterData build_inmemory_cluster_data(
    std::vector<float>& raw_data, size_t n, size_t d, Kmeans& kmeans)
{
  InMemoryClusterData data;
  data.n = n;
  data.d = d;
  data.cluster_num = kmeans.cluster_num;
  data.centroids = kmeans.centroids_;
  data.assignment = std::move(kmeans.inverted_list_);
  data.bucket_sizes.resize(data.cluster_num);
  data.file_pos.resize(data.cluster_num);
  data.radii.resize(data.cluster_num);

  size_t cumu_points = 0;
  for (size_t i = 0; i < data.cluster_num; i++)
  {
    auto& cluster = data.assignment[i];
    std::sort(cluster.begin(), cluster.end());
    data.bucket_sizes[i] = cluster.size();
    data.file_pos[i] = cumu_points * d * sizeof(float);
    data.max_points = std::max(data.max_points, cluster.size());
    cumu_points += cluster.size();
  }

  data.cluster_data.resize(n * d);
  for (size_t i = 0; i < data.cluster_num; i++)
  {
    float* cluster_ptr =
        data.cluster_data.data() + data.file_pos[i] / sizeof(float);
    float radius = 0;
    for (size_t j = 0; j < data.assignment[i].size(); j++)
    {
      size_t id = data.assignment[i][j];
      float* src = raw_data.data() + id * d;
      memcpy(cluster_ptr + j * d, src, d * sizeof(float));
      float dist = dist_l2(src, data.centroids.data() + i * d, &d);
      if (dist > radius) radius = dist;
    }
    data.radii[i] = std::sqrt(radius);
  }

  raw_data.clear();
  raw_data.shrink_to_fit();
  return data;
}

inline InMemoryBuildArtifacts one_level_kmeans_inmemory(ConfigReader config)
{
  size_t n = 0;
  size_t d = 0;
  std::vector<float> raw_data = load_all_data(config.data_file, n, d);
  if (config.unit_normalize_prebuild)
  {
    normalize_vectors_in_place(raw_data.data(), n, d);
  }
  std::vector<float> centroids =
      sample_centroids_from_memory(raw_data, n, d, config.cluster_num);

  Kmeans kmeans(d, config.cluster_num,
                kmeans_objective_from_string(config.partition_objective),
                config.eta);
  kmeans.centroids_ = centroids;

  std::unique_ptr<hnswlib::L2Space> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;

  if (config.partition_objective == "standard")
  {
    space.reset(new hnswlib::L2Space(d));
    graph.reset(new hnswlib::HierarchicalNSW<float>(
        space.get(), config.cluster_num, 20, 100));
#pragma omp parallel for
    for (size_t i = 0; i < config.cluster_num; i++)
    {
      graph->addPoint(centroids.data() + i * d, i);
    }
  }

  size_t batch_size = std::max<size_t>(1, n / 1000);
  std::vector<size_t> ids(batch_size);
  for (size_t i = 0; i < div_round_up(n, batch_size); i++)
  {
    size_t num = batch_size * (i + 1) < n ? batch_size : n - batch_size * i;
    for (size_t j = 0; j < num; j++)
    {
      ids[j] = i * batch_size + j;
    }
    if (config.partition_objective == "anisotropic")
    {
      kmeans.add_anisotropic_exact(num, raw_data.data() + i * batch_size * d,
                                   ids);
    }
    else
    {
      kmeans.add2choice(num, raw_data.data() + i * batch_size * d, ids,
                        *graph);
    }
  }

  InMemoryBuildArtifacts artifacts;
  if (config.partition_objective == "anisotropic")
  {
    verify_anisotropic_assignments_memory(kmeans, raw_data.data(), n, d,
                                          kmeans.inverted_list_);
  }
  artifacts.cluster_data = build_inmemory_cluster_data(raw_data, n, d, kmeans);
  if (graph)
  {
    graph->radii = artifacts.cluster_data.radii;
  }
  artifacts.space = std::move(space);
  artifacts.graph = std::move(graph);
  return artifacts;
}

struct InMemoryDiskRange
{
  InMemoryClusterData cluster_data_;
  std::unique_ptr<hnswlib::L2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph_;

  void build(ConfigReader config)
  {
    InMemoryBuildArtifacts artifacts = one_level_kmeans_inmemory(config);
    cluster_data_ = std::move(artifacts.cluster_data);
    space_ = std::move(artifacts.space);
    graph_ = std::move(artifacts.graph);
  }

  void search(ConfigReader config)
  {
    if (!graph_)
    {
      throw std::runtime_error(
          "anisotropic in-memory build does not produce a search graph");
    }
    float epsilon = std::sqrt(config.radius);
    size_t cluster_num = cluster_data_.cluster_num;
    size_t d = cluster_data_.d;
    auto& bucket_sizes = cluster_data_.bucket_sizes;
    auto& assignment = cluster_data_.assignment;
    float* centroids = cluster_data_.centroids.data();
    std::vector<std::vector<size_t>> tasks(cluster_num);

    graph_->radii = cluster_data_.radii;
    graph_->setEf(1000);

    int len = 500;
    std::vector<float> arcos_list(len + 1);
    float sc = len / 2;
    for (int i = 0; i <= len; i++)
    {
      float x = float(i - sc) / sc;
      arcos_list[i] = std::acos(x);
    }

    auto arcos = [&](float x) -> float
    {
      if (x > 1) x = 1;
      int index = x * len / 2 + len / 2;
      return arcos_list[index];
    };

#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; i++)
    {
      auto top_candidates = graph_->knnSearchBaseLayer(i, config.K);
      std::vector<std::pair<float, unsigned>> dis_to_boundary;
      while (!top_candidates.empty())
      {
        auto neighbor_cluster = top_candidates.top().second;
        float temp = top_candidates.top().first / 2;
        top_candidates.pop();
        dis_to_boundary.emplace_back(temp, neighbor_cluster);
      }
      std::sort(dis_to_boundary.begin(), dis_to_boundary.end());
      int ptr = dis_to_boundary.size() - 1;
      float sum_of_angle = 0;
      while (ptr >= 0)
      {
        sum_of_angle += arcos(dis_to_boundary[ptr].first / epsilon);
        if (sum_of_angle >= config.error_bound) break;
        ptr--;
      }
      for (int ii = 0; ii <= ptr; ii++)
      {
        if (i <= dis_to_boundary[ii].second)
        {
          tasks[i].push_back(dis_to_boundary[ii].second);
        }
      }
      std::sort(tasks[i].begin(), tasks[i].end());
      if (tasks[i].size() == 0)
      {
        for (int ii = 0; ii <= ptr; ii++)
        {
          if (dis_to_boundary[ii].first != 0) exit(0);
        }
        tasks[i].push_back(i);
      }
    }

    std::vector<size_t> perm(cluster_num);
    std::vector<size_t> order(cluster_num);
    std::vector<std::vector<size_t>> reordered_tasks(cluster_num);
    std::vector<std::vector<size_t>> shuffled_tasks(cluster_num);
    for (size_t i = 0; i < cluster_num; i++)
    {
      perm[i] = i;
      shuffled_tasks[i] = tasks[i];
    }

    size_t window = config.gorder_window == 0 ? 1 : config.gorder_window;
    perm = order_gorder(shuffled_tasks, window);
    for (size_t i = 0; i < cluster_num; i++)
    {
      reordered_tasks[perm[i]] = shuffled_tasks[i];
      order[perm[i]] = i;
    }

    size_t count = 0;
    float dist_comp = 0;
    for (size_t i = 0; i < cluster_num; i++)
    {
      auto target_cluster = order[i];
      auto& target_tasks = reordered_tasks[i];
      if (target_tasks.empty() || bucket_sizes[target_cluster] == 0) continue;

      std::vector<const float*> task_data(target_tasks.size());
      for (size_t j = 0; j < target_tasks.size(); j++)
      {
        task_data[j] = cluster_data_.cluster_data.data() +
                       cluster_data_.file_pos[target_tasks[j]] / sizeof(float);
      }

#pragma omp parallel for schedule(dynamic) reduction(+ : count) \
    reduction(+ : dist_comp)
      for (size_t j = 0; j < bucket_sizes[target_cluster]; j++)
      {
        auto id1 = assignment[target_cluster][j];
        const float* vec1 = task_data[0] + j * d;
        for (size_t l = 0; l < bucket_sizes[target_cluster]; l++)
        {
          auto id2 = assignment[target_cluster][l];
          if (id2 >= id1) continue;
          const float* vec2 = task_data[0] + l * d;
          float dist = dist_l2(vec1, vec2, &d);
          dist_comp++;
          if (dist < epsilon * epsilon)
          {
            if (id1 % 100000 == 0) count++;
            if (id2 % 100000 == 0) count++;
          }
        }

        for (size_t k = 1; k < target_tasks.size(); k++)
        {
          auto neighbor_cluster = target_tasks[k];
          for (size_t l = 0; l < bucket_sizes[neighbor_cluster]; l++)
          {
            auto id2 = assignment[neighbor_cluster][l];
            const float* vec2 = task_data[k] + l * d;
            float dist = dist_l2(vec1, vec2, &d);
            dist_comp++;
            if (dist < epsilon * epsilon)
            {
              if (id1 % 100000 == 0) count++;
              if (id2 % 100000 == 0) count++;
            }
          }
        }
      }
    }

    std::cout << "recall = " << 1.0 * count / config.gt << "\n";
  }
};

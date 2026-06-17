#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/ConfigLoader.h"

namespace fs = std::filesystem;

namespace
{
void require(bool ok, const std::string& message)
{
  if (!ok)
  {
    throw std::runtime_error(message);
  }
}

std::string make_temp_dir()
{
  fs::path base = fs::temp_directory_path() / "approx_alpha_config_test";
  fs::remove_all(base);
  fs::create_directories(base);
  return base.string();
}

fs::path copy_config_with_tail(const std::string& base_config,
                               const std::string& temp_dir,
                               const std::string& name,
                               const std::vector<std::string>& tail_lines)
{
  const fs::path path = fs::path(temp_dir) / name;
  std::ifstream in(base_config);
  require(static_cast<bool>(in), "failed to open base config");
  std::ofstream out(path);
  require(static_cast<bool>(out), "failed to create temp config");
  out << in.rdbuf();
  if (!tail_lines.empty())
  {
    out << '\n';
  }
  for (const std::string& line : tail_lines)
  {
    out << line << '\n';
  }
  return path;
}

void expect_alpha(const std::string& config_path, const std::string& gt_path,
                  float expected)
{
  const ResolvedConfig config = load_resolved_config(config_path, gt_path);
  require(std::fabs(config.approx_alpha - expected) <= 1e-6f,
          "unexpected approx_alpha value");
}

void expect_rejected(const std::string& config_path, const std::string& gt_path,
                     const std::string& value_label)
{
  bool rejected = false;
  try
  {
    (void)load_resolved_config(config_path, gt_path);
  }
  catch (const std::exception& e)
  {
    rejected = std::string(e.what()).find("approx_alpha") != std::string::npos;
  }
  require(rejected, "approx_alpha config must reject " + value_label);
}
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "usage: approx_alpha_config_test <config> <gt>\n";
    return 2;
  }

  try
  {
    const std::string base_config = argv[1];
    const std::string gt_path = argv[2];
    const std::string temp_dir = make_temp_dir();

    expect_alpha(base_config, gt_path, 1.0f);

    const fs::path explicit_alpha =
        copy_config_with_tail(base_config, temp_dir, "explicit_alpha.config",
                              {"approx_alpha 1.5"});
    expect_alpha(explicit_alpha.string(), gt_path, 1.5f);

    const std::vector<std::pair<std::string, std::string>> invalid_cases = {
        {"approx_alpha 0.5", "sub-1.0"},
        {"approx_alpha nan", "nan"},
        {"approx_alpha inf", "inf"},
        {"approx_alpha abc", "non-numeric"},
        {"approx_alpha 1.5junk", "trailing-token-garbage"},
        {"approx_alpha 1.0#bad", "hash-token-garbage"},
        {"approx_alpha 1.5 junk", "extra-token"},
        {"approx_alpha 1.0 #bad", "extra-comment-token"},
    };
    for (size_t i = 0; i < invalid_cases.size(); ++i)
    {
      const fs::path path = copy_config_with_tail(
          base_config, temp_dir, "invalid_" + std::to_string(i) + ".config",
          {invalid_cases[i].first});
      expect_rejected(path.string(), gt_path, invalid_cases[i].second);
    }

    const fs::path duplicate = copy_config_with_tail(
        base_config, temp_dir, "duplicate_alpha.config",
        {"approx_alpha 1.2", "approx_alpha 1.5"});
    expect_rejected(duplicate.string(), gt_path, "duplicate key");

    fs::remove_all(temp_dir);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}

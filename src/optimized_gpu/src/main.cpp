#include "raster.h"
#include "sunshine_hours.h"
#include "timer.h"
#include "config.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cstring>
#include <filesystem>
#include <thread>
#include <chrono>

// ===== Global configuration parameter definitions =====
int BATCH         = 16;           // Default value; can be overridden by configuration
int NUM_STREAMS   = 16;
dim3 BLOCK_SIZE   = dim3(64, 4);
int CUDA_DEVICE_ID = 0;

// ===== Sunshine calculation defaults =====
int DAY_OF_YEAR     = 15;
int TIME_STEP       = 5;
float PADDING_DEGREE = 1.0f;
std::string FILE_PATH = "";
std::string OUTPUT_PATH = "";
// ===== End =====


static std::string make_output_path(const std::string& input_path,
                                    const std::string& out_root,
                                    int day)
{
    namespace fs = std::filesystem;
    fs::path in_path(input_path);
    std::string stem = in_path.stem().string();       // File name without extension
    fs::path dir = fs::path(out_root) / stem;         // <out_root>/<stem>

    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    };

    std::string filename = stem + "--_result" + std::to_string(day) + in_path.extension().string();
    return (dir / filename).string();
}

// Failure log file name
static const char* FAIL_LOG_NAME = "faillog.txt";

void parse_args(int argc, char* argv[])
{
    auto need = [&](int& i, const char* name){
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << name << '\n';
            std::exit(1);
        }
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg=="--file"||arg=="-f"){ need(i,"--file"); FILE_PATH=argv[++i]; }
        else if(arg=="--gpu"||arg=="-g"){ need(i,"--gpu"); CUDA_DEVICE_ID=std::stoi(argv[++i]); }
        else if(arg=="--batch"){ need(i,"--batch"); BATCH=std::stoi(argv[++i]); }
        else if(arg=="--streams"){ need(i,"--streams"); NUM_STREAMS=std::stoi(argv[++i]); }
        else if (arg == "--blockx" || arg == "-bx") {
            need(i, "--blockx");
            BLOCK_SIZE.x = std::stoi(argv[++i]);
        }
        else if (arg == "--blocky" || arg == "-by") {
            need(i, "--blocky");
            BLOCK_SIZE.y = std::stoi(argv[++i]);
        }
        else if(arg=="--day"){ need(i,"--day"); DAY_OF_YEAR=std::stoi(argv[++i]); }
        else if(arg=="--step"){ need(i,"--step"); TIME_STEP=std::stoi(argv[++i]); }
        else if(arg=="--pad"){ need(i,"--pad"); PADDING_DEGREE=std::stof(argv[++i]); }
        else if(arg=="--output"||arg=="-o"){ need(i,"--output"); OUTPUT_PATH=argv[++i]; }
        else if(arg=="--help"||arg=="-h"){
            std::cout<<"Usage: ./sunshine_hours [options]\n"
                     <<"  -f|--file <DEM>\n  -g|--gpu <id>\n"
                     <<"  --batch <rows>\n  --streams <N>\n"
                     <<"  --blockx|-bx <bx>\n"
                     <<"  --blocky|-by <by>\n"
                     <<"  --day <doy>\n"
                     <<"  --step <min>\n  --pad <deg>\n  --output <path>";
            std::exit(0);
        }
    }
}

int main(int argc, char* argv[])
{
    // Parse command-line arguments
    parse_args(argc, argv);

    timer.tick("total");
    std::string dem_file_path = FILE_PATH;
    std::string result_root  = OUTPUT_PATH;   // If no suffix is present, treat it as an output root directory
    cudaSetDevice(CUDA_DEVICE_ID);

    // If OUTPUT_PATH is a directory, construct the output subdirectory and file name automatically
    {
        namespace fs = std::filesystem;
        fs::path outp(result_root);
        if (!outp.has_extension()) {
            if (DAY_OF_YEAR == 365) {
                // Multi-day output: create <out_root>/<stem> directory
                result_root = (outp / fs::path(dem_file_path).stem()).string();
            } else {
                // Single-day output: full file path
                result_root = make_output_path(dem_file_path, outp.string(), DAY_OF_YEAR);
            }
        }
    }

    // ---------- Determine the faillog write path ----------
    std::filesystem::path out_root_path(OUTPUT_PATH);
    std::filesystem::path fail_dir;
    if (!OUTPUT_PATH.empty() && !out_root_path.has_extension()) {
        // User provided a directory
        fail_dir = out_root_path;
    } else {
        // User provided a file path, or nothing; empty means current directory
        std::filesystem::path rp(result_root);
        fail_dir = rp.has_extension() ? rp.parent_path() : rp;
    }
    if (!std::filesystem::exists(fail_dir) && !fail_dir.empty()) {
        std::filesystem::create_directories(fail_dir);
    }
    std::string fail_log_path = (fail_dir / FAIL_LOG_NAME).string();

    auto print_first_cells = [](const Raster &r, int n, const std::string &name) {
        std::cout << "---- First " << n << " cells of " << name << " ----\n";
        if (r.size() == 0) {
            std::cout << "(empty raster)\n";
            return;
        }
        const double* G = r.geo_transform;
        int printed = 0;
        for (size_t i = 0; i < r.rows && printed < n; ++i) {
            for (size_t j = 0; j < r.cols && printed < n; ++j) {
                // Pixel-center coordinates
                double x = G[0]
                         + j * G[1]
                         + i * G[2]
                         + 0.5 * G[1]
                         + 0.5 * G[2];
                double y = G[3]
                         + j * G[4]
                         + i * G[5]
                         + 0.5 * G[4]
                         + 0.5 * G[5];

                float v = r.data.get()[i * r.cols + j];
                std::cout << name << "[" << i << "," << j << "] = "
                          << v << " at (" << x << ", " << y << ")\n";
                ++printed;
            }
        }
    };

    // Read the DEM with retry logic, up to five attempts
    const int DEM_MAX_RETRIES = 5;  // Maximum DEM read retry count
    Raster dem;                     // Declared before the loop for later use
    bool dem_loaded = false;
    for (int attempt = 1; attempt <= DEM_MAX_RETRIES && !dem_loaded; ++attempt) {
        std::cout << "\n=== Reading DEM (attempt " << attempt << ") ===\n";
        timer.tick("Reading DEM");
        try {
            dem = Raster(dem_file_path);
            if (dem.size() == 0)
                throw std::runtime_error("DEM size is zero");
            dem.printInfo();
            // print_first_cells(dem, 20, "DEM");
            timer.tock();              // Record elapsed time for this read attempt
            dem_loaded = true;         // Mark success
        }
        catch (const std::exception &e) {
            timer.tock();
            std::cerr << "[WARN] Reading DEM attempt " << attempt
                      << " failed: " << e.what() << "\n";
            if (attempt < DEM_MAX_RETRIES) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
        catch (...) {
            timer.tock();
            std::cerr << "[WARN] Reading DEM attempt " << attempt
                      << " failed with unknown error.\n";
            if (attempt < DEM_MAX_RETRIES) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
    }

    if (!dem_loaded) {
        std::cerr << "[ERR] Failed to read DEM after "
                  << DEM_MAX_RETRIES << " attempts.\n";
        // Log the failed file; all means the whole file failed
        {
            std::ofstream flog(fail_log_path, std::ios::app);
            if (flog) {
                flog << dem_file_path << " all\n";
            }
        }
        return 1;  // Terminate the program
    }

    std::cout << "block" << BLOCK_SIZE.x << " " << BLOCK_SIZE.y << std::endl;
 
    if (DAY_OF_YEAR == 365) {
        const int MAX_RETRIES = 3;  // Maximum retry count per day
        std::filesystem::create_directories(result_root);
        std::vector<int> failed_days;   // Collect failed days
        for (int doy = 1; doy <= 365; ++doy) {
            bool success = false;
            SunshineHours sh(dem, PADDING_DEGREE, doy, TIME_STEP);

            for (int attempt = 1; attempt <= MAX_RETRIES && !success; ++attempt) {
                std::cout << "\n=== Calculate day " << doy 
                        << " (attempt " << attempt << ") ===\n";

                // Time and retry only the calculate() section
                timer.tick("Sh.calculate");  
                try {
                    sh.calculate();
                    timer.tock();  // Record elapsed time for this calculation attempt
                    success = true;              // Mark success and exit retry loop
                }
                catch (const std::exception& e) {
                    timer.tock();  // Record elapsed time for this failed attempt
                    std::cerr << "[WARN] Day " << doy 
                            << " attempt " << attempt 
                            << " calculate() failed: " << e.what() << "\n";
                    if (attempt < MAX_RETRIES) {
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    }
                }
                catch (...) {
                    timer.tock();
                    std::cerr << "[WARN] Day " << doy 
                            << " attempt " << attempt 
                            << " calculate() failed with unknown error.\n";
                    if (attempt < MAX_RETRIES) {
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    }
                }
            }

            if (!success) {
                failed_days.push_back(doy);
                continue;
            }

            // Save the result only after a successful calculation
            timer.tick("SaveResult");
            std::string out_file = result_root + "/day" 
                                + std::to_string(doy) + ".tif";
            Raster res = sh.getResult();

            // print_first_cells(res, 20, "Result day " + std::to_string(doy));

            res.save(out_file);
            timer.tock();
        }

        // Batch-write failed days, if any
        if (!failed_days.empty()) {
            std::ofstream flog(fail_log_path, std::ios::app);
            if (flog) {
                flog << dem_file_path << " ";
                for (size_t i = 0; i < failed_days.size(); ++i) {
                    if (i) flog << ",";
                    flog << failed_days[i];
                }
                flog << "\n";
            }
        }
    }
    else {
        // Single-day mode: time calculate and save directly
        timer.tick("Sh.calculate");
        SunshineHours sunshine_hours(dem, PADDING_DEGREE, DAY_OF_YEAR, TIME_STEP);
        sunshine_hours.calculate();
        timer.tock();

        timer.tick("SaveResult");
        Raster result = sunshine_hours.getResult();

        // print_first_cells(result, 20, "Result day " + std::to_string(DAY_OF_YEAR));
        
        result.save(result_root);
        timer.tock();
    }

    timer.tock();   // End total timing
    timer.print_records();

    // result.print_info();
    cudaDeviceReset();

    return 0;
}
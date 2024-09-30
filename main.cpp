#include <fftw3.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <complex>
#include <algorithm>
using namespace std;
// g++ -o main main.cpp -I$HOME/fftw/include -L$HOME/fftw/lib -lfftw3
double range, bound;
int width, height, depth, size2;
double PI = 3.14159265358979323846;
std::vector<std::complex<double>> compute_fft(std::vector<double>& input_data) {
    int N = input_data.size();  

    
    fftw_complex* output = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, input_data.data(), output, FFTW_ESTIMATE);

 
    fftw_execute(plan);

    
    std::vector<std::complex<double>> fft_result(N);
    for (int i = 0; i < N; ++i) {
        fft_result[i] = std::complex<double>(output[i][0], output[i][1]);
    }

    
    fftw_destroy_plan(plan);
    fftw_free(output);

    return fft_result;
}


std::vector<double> getdata(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::vector<double> data;
    if (!file) {
        std::cerr << "can not open file" << std::endl;
        return data;
    }

    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    
    std::streamsize num_floats = size / sizeof(double);
    

    std::vector<double> buffer(num_floats);

    
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        
        return buffer;
    } else {
        std::cerr << "can not read file" << std::endl;
        return buffer;
    }
    
    return buffer;
}

std::vector<double> generate_gaussian_noise(int n, double m, double f_min, double f_max, double fs, unsigned int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0); // mean = 0, std = 1

    std::vector<double> noise(n);
    double t_step = 1.0 / fs;

    for (int i = 0; i < n; ++i) {
        double t = i * t_step;
        double freq = f_min + (f_max - f_min) * distribution(generator); // 随机选择频率在 [f_min, f_max] 范围
        noise[i] = m * std::sin(2 * M_PI * freq * t);  // 生成带有正态分布的随机噪声
    }
    
    return noise;
}

double max_abs_value(const std::vector<std::complex<double>>& vec) {
    double max_val = 0.0;
    for (const auto& val : vec) {
        double abs_val = std::abs(val);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

// 对向量进行归一化处理
std::vector<std::complex<double>> normalize_vector(const std::vector<std::complex<double>>& vec) {
    double max_val = max_abs_value(vec);
    std::vector<std::complex<double>> normalized_vec(vec.size());
    
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized_vec[i] = vec[i] / max_val;  // 归一化
    }

    return normalized_vec;
}

// 计算两个归一化后的向量之间的加权误差
double compute_weighted_error(const std::vector<std::complex<double>>& fft_result,
                              const std::vector<std::complex<double>>& decp_fft_result,
                              double a, double b, double f_0, double epsilon, double sigma,
                              double W_inside, double W_outside, int N) {
    // 归一化两个向量
    std::vector<std::complex<double>> norm_fft_result = normalize_vector(fft_result);
    std::vector<std::complex<double>> norm_decp_fft_result = normalize_vector(decp_fft_result);

    double weighted_error = 0.0;
    for (int i = 0; i < N; ++i) {
        // 计算归一化频率值 (假设采样频率为 1)
        double freq = static_cast<double>(i) / N;  // 归一化频率范围为 [0, 0.5]

        // 计算归一化后的 FFT 差异 (取绝对值)
        double diff = std::abs(norm_fft_result[i] - norm_decp_fft_result[i]);

        // 定义权重 (频段 [a, b] 内使用倒数频率加权，频段外使用高斯衰减)
        double weight;
        if (freq >= a && freq <= b) {
            weight = W_inside / (freq + epsilon);  // 倒数频率加权，确保低频部分权重大
        } else {
            weight = W_outside * std::exp(-std::pow(freq - f_0, 2) / (2 * std::pow(sigma, 2)));  // 高斯衰减
        }

        // 加权误差累加
        weighted_error += diff * weight;
    }

    return weighted_error;
}

std::vector<double> add_vectors(const std::vector<double>& vec1, const std::vector<double>& vec2) {
   
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition.");
    }

    
    std::vector<double> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

double max_absolute_difference(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    // 检查向量大小是否相同
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size.");
    }

    // 初始化一个变量来存储最大绝对差值
    double max_diff = 0.0;

    // 逐元素计算差值的绝对值，并找出最大值
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = std::abs(vec1[i] - vec2[i]);  // 计算绝对差值
        if (diff > max_diff) {
            max_diff = diff;  // 更新最大差值
        }
    }

    return max_diff;
}

int main(int argc, char** argv) {
    std::string command;
    std::string dimension = argv[1];
    range = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int result;
    std::string input_filename, filename;
    std::istringstream iss(dimension);
    char delimiter;
    if (std::getline(iss, filename, ',')) {
        if (iss >> width >> delimiter && delimiter == ',' &&
            iss >> height >> delimiter && delimiter == ',' &&
            iss >> depth) {
            std::cout << "Filename: " << filename << std::endl;
            std::cout << "Width: " << width << std::endl;
            std::cout << "Height: " << height << std::endl;
            std::cout << "Depth: " << depth << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for filename" << std::endl;
    }

    // filename="signal";
    // get input/decompressed data;
    input_filename = filename+".bin";
    size2 = width * height * depth;
    std::vector<double> input_data, decp_data;
    input_data = getdata(input_filename);

    auto min_it = std::min_element(input_data.begin(), input_data.end());
    auto max_it = std::max_element(input_data.begin(), input_data.end());
    double minValue = *min_it;
    double maxValue = *max_it;


    bound = (maxValue-minValue)*range;
    
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6) << range;


    std::ostringstream oss1;
    oss1 << std::scientific << std::setprecision(17) << bound;

    std::string decp_filename = "/pscratch/sd/y/yuxiaoli/fft/decompressed_data/decp_"+filename+"_"+oss.str()+compressor_id+"_.bin";
    std::string cpfilename = "/pscratch/sd/y/yuxiaoli/fft/compressed_data/compressed_"+filename+"_"+oss.str()+".sz";
    // std::string fix_path = "/pscratch/sd/y/yuxiaoli/MSCz/fixed_decp_data/fixed_decp_"+filename+"_"+oss.str()+".bin";
    cout<<"decp_filename: "<<decp_filename<<endl;
    if(compressor_id=="sz3"){
        
        command = "sz3 -i " + input_filename + " -z " + cpfilename +" -o "+ decp_filename + " -d " + " -1 " + std::to_string(size2) + " -M "+"REL "+oss.str()+" -a";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    else if(compressor_id=="zfp"){
        cpfilename = "/pscratch/sd/y/yuxiaoli/fft/compressed_data/compressed_"+filename+"_"+oss.str()+".zfp";
        // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
        // decp_filename = "/pscratch/sd/y/yuxiaoli/MSCz/decompressed_data/decp_"+filename+"_"+std::to_string(bound1)+compressor_id+"_.bin";
        
        command = "zfp -i " + input_filename + " -z " + cpfilename +" -o "+decp_filename + " -d " + " -1 " + std::to_string(size2)+" -a "+oss1.str()+" -s";
        std::cout << "Executing command: " << command << std::endl;
       
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    decp_data = getdata(decp_filename);

    std::vector<std::complex<double>> fft_result = compute_fft(input_data);
    std::vector<std::complex<double>> decp_fft_result = compute_fft(decp_data);

    double a = 0.04;  
    double b = 0.06;  
    double f_0 = (a + b) / 2; 
    double epsilon = 1e-10; 
    double sigma = 0.01; 
    double W_inside = 1.0;  
    double W_outside = 0.05; 

    
    int N = fft_result.size();

    
    double weighted_error = compute_weighted_error(fft_result, decp_fft_result, a, b, f_0, epsilon, sigma, W_inside, W_outside, N);

    int n = size2;
    double m = bound/2.0;
    double f_min = 0.04;
    double f_max = 0.06;
    double fs = 2*f_max;
    unsigned int seed = 42;

    std::vector<double> noise = generate_gaussian_noise(n, m, f_min, f_max, fs, seed);
    
    std::cout << "Weighted sum of error: " << weighted_error << std::endl;
    while(weighted_error>1.0)
    {
        range /= 2;
        bound /= 2;
        m = bound/32;
        oss << std::scientific << std::setprecision(6) << range;
        oss1 << std::scientific << std::setprecision(17) << bound;
        decp_filename = "/pscratch/sd/y/yuxiaoli/fft/decompressed_data/decp_"+filename+"_"+oss.str()+compressor_id+"_.bin";
        cpfilename = "/pscratch/sd/y/yuxiaoli/fft/compressed_data/compressed_"+filename+"_"+oss.str()+".sz";
        
        if(compressor_id=="sz3"){
            
            command = "sz3 -i " + input_filename + " -z " + cpfilename +" -o "+ decp_filename + " -d " + " -1 " + std::to_string(size2) + " -M "+"REL "+oss.str()+" -a";
            std::cout << "Executing command: " << command << std::endl;
            result = std::system(command.c_str());
            if (result == 0) {
                std::cout << "Compression successful." << std::endl;
            } else {
                std::cout << "Compression failed." << std::endl;
            }
        }
        else if(compressor_id=="zfp"){
            cpfilename = "/pscratch/sd/y/yuxiaoli/fft/compressed_data/compressed_"+filename+"_"+oss.str()+".zfp";
            // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
            // decp_filename = "/pscratch/sd/y/yuxiaoli/MSCz/decompressed_data/decp_"+filename+"_"+std::to_string(bound1)+compressor_id+"_.bin";
            
            command = "zfp -i " + input_filename + " -z " + cpfilename +" -o "+decp_filename + " -d " + " -1 " + std::to_string(size2)+" -a "+oss1.str()+" -s";
            std::cout << "Executing command: " << command << std::endl;
        
            result = std::system(command.c_str());
            if (result == 0) {
                std::cout << "Compression successful." << std::endl;
            } else {
                std::cout << "Compression failed." << std::endl;
            }
        }

        noise = generate_gaussian_noise(n, m, f_min, f_max, fs, seed);
        decp_data = getdata(decp_filename);
        decp_data = add_vectors(decp_data, noise);
        decp_fft_result = compute_fft(decp_data);
        weighted_error = compute_weighted_error(fft_result, decp_fft_result, a, b, f_0, epsilon, sigma, W_inside, W_outside, N);
        std::cout << "Weighted sum of error: " << weighted_error << std::endl;
    }

    double max_diff = max_absolute_difference(decp_data, input_data);

    std::cout << "Maximum absolute difference: " << max_diff << std::endl;
    std::cout << "REL error: " << range << std::endl;
    std::cout << "ABS error: " << bound << std::endl;
    return 0;
}

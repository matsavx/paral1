#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>
#include <atomic>
//goldbolt

#define  STEPS 10000000
#define CACHE_LINE 64u

using namespace std;

typedef double (* f_t)(double);
double integrate_cpp_mtx(double a, double b, f_t f) {
    using namespace std;
    unsigned T = thread::hardware_concurrency();
    vector<thread> threads;
    mutex mtx;
    double result = 0, dx = (b - a) / STEPS;
    for (unsigned t = 0; t < T; t++)
        threads.emplace_back([=, & result, & mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T)
                R += f(i * dx + a);
//            omp_set_lock lck{mtx};
            result += R;
        });
    for (auto&thr:threads)
        thr.join();
    return result*dx;
}
double func (double x) {
    return x*x;
}

struct partial_sum_t {
    alignas(64) double value;
};

double integrate_partial_sum (double a, double b, double (*f)(double)) {
	double global_result = 0;
	partial_sum_t *partial_sum;
	double dx = (b - a) / STEPS;
	unsigned T;

#pragma omp parallel shared (partial_sum, T)
	{
		unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
		{
			T = (unsigned) omp_get_num_threads();
			partial_sum = (partial_sum_t *) aligned_alloc(T * sizeof(partial_sum_t), CACHE_LINE);
			partial_sum[t].value = 0;
			for (unsigned int i = t; i < STEPS; i += T) {
				partial_sum[t].value += f(i * dx + a);
			}
			for (unsigned int i = 0; i < T; i++) {
				global_result += partial_sum[i].value;
			}
		};
	};
	return global_result * dx;
}

double integrate_reduction (double a, double b, double (*f)(double)) {
    double result = 0.0;
    double dx = (b -a)/STEPS;
#pragma omp parallel for reduction(+:result)
    for(int i = 0; i < STEPS; i++)
        result += f(a+dx*i);
    return result;
}

double integratePS (double a, double b, f_t f) {
    double dx = (b-a)/STEPS;
    double result = 0;
    unsigned T = thread::hardware_concurrency();
    auto vec = vector(T, partial_sum_t{0.0});
    vector<thread> thread_vec;
    auto thread_proc = [&vec, dx, T, f, a](auto t) {
        for (unsigned i = t; i < STEPS; i+=T) {
            vec[t].value += f(dx*i + a);
        }
    };
    for (unsigned t = 1; t < T; t++)
        thread_vec.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread:thread_vec)
        thread.join();
    for (auto elem:vec)
        result = elem.value;
    return result*dx;
}
double Quadratic(double x)
{
    return x*x;
}
typedef double (*function)(double);
typedef struct experiment_result_t_{
    double result, time_ms;
} experiment_result_t;
typedef double (*I_t) (double, double, function);
experiment_result_t run_experiment(I_t I) {
    double t0 = omp_get_wtime();
    double res = I(-1, 1, Quadratic);
    double t1 = omp_get_wtime();

    experiment_result_t Result;
    Result.result = res;
    Result.time_ms = t1 - t0;

    return Result;
}

void show_experiment_results(I_t I) {
    printf("%10s %10sms\n", "Result", "Time");
    for (unsigned T = 1; T <= omp_get_num_procs(); T++) {
        experiment_result_t R;
        omp_set_num_threads(T);
        R = run_experiment(I);
        printf("%10g%10g\n", R.result, R.time_ms);
    }
}

double integrate(double a, double b, f_t f) {
    atomic<double> Result {0.0};
    int num = omp_get_num_threads();
    vector<thread> threads;
    double dx = (b - a)/STEPS;
    auto fun = [dx,&Result,a,b,f, num](auto t) {
        for(unsigned i = t; i < STEPS; i += num) {
            Result = Result + f(i*dx+a);
        }
    };
    for (unsigned int t = 1; t < num; t++)
        threads.emplace_back(fun,t);
    fun(0);
    for (auto &thread:threads)
        thread.join();
    return Result * dx;
}

int main() {
//    cout << integrate(-1,1,func) << endl;
//    std::cout << integratePS(-1,1,func)<<std::endl;
	std::cout << "integrate\n";
	show_experiment_results(integrate);
	std::cout << "integrate_cpp_mtx\n";
    show_experiment_results(integrate_cpp_mtx);
	std::cout << "integrate_partial_sum\n";
	show_experiment_results(integratePS);
	std::cout << "integrate_partial_sum_omp\n";
	show_experiment_results(integrate_partial_sum);
	std::cout << "integrate_reduction\n";
	show_experiment_results(integrate_reduction);
//    experiment_result_t ExpP = run_experiment(integratePS);
}

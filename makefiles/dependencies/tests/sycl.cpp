#include <CL/sycl.hpp>

int main() {
    try {
        sycl::queue q{ sycl::default_selector_v };
        auto dev = q.get_device();        
    } 
    catch (const sycl::exception &e) {
        std::cerr << "SYCL error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#ifndef LYAPUNOV_CUH
#define LYAPUNOV_CUH

#include <Fractal/fractal.cuh>

class lyapunov : public fractal {
public:
    lyapunov();
    void generate(const FractalParams& params) override;
};

#endif

#ifndef BARNSLEY_CUH
#define BARNSLEY_CUH

#include <Fractal/fractal.cuh>

class barnsley : public fractal {
public:
    barnsley();
    void generate(const FractalParams& params) override;
};

#endif

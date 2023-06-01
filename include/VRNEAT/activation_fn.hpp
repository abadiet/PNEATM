#ifndef ACTIVATION_FN_HPP
#define ACTIVATION_FN_HPP

#include <functional>

namespace vrneat {

class ActivationFn{
    public:
        ActivationFn (int in_kind, int out_kind, void* func (void* input));
        void* use (void* input);

    private:
        int id;
        int in_kind, out_kind;
        std::function<void* (void*)> func;
};

}

#endif	// ACTIVATION_FN_HPP
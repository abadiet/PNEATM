#include <VRNEAT/activation_fn.hpp>

using namespace vrneat;

ActivationFn::ActivationFn (int in_kind, int out_kind, void* func (void* input)) : in_kind (in_kind), out_kind (out_kind), func (func) {

}

void* ActivationFn::use (void* input) {
    return func (input);
}
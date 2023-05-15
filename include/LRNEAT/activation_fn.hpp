#ifndef ACTIVATION_FN_HPP
#define ACTIVATION_FN_HPP

namespace neat {

class ActivationFn{
    public:
        ActivationFn (int in_kind, int out_kind, void* func (void* input));
        use (void* input);

    private:
        int id;
        int in_kind, out_kind;
        void* func (void* input);
};

}

#endif	// ACTIVATION_FN_HPP
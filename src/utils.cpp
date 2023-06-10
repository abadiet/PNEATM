#include <PNEATM/utils.hpp>

using namespace pneatm;

template <typename T1, typename T2, typename... Args>
struct CreateNode {
    static NodeBase* get(size_t iT_in, size_t iT_out, bool T2_is_first = false, bool mono_type = true) {
        if (iT_in == 0 && iT_out == 0) {
            if (mono_type) {
                return std::make_unique<Node <T1, T1> ()>;
            }
            if (T2_is_first) {
                return std::make_unique<Node <T2, T1> ()>;
            } else {
                return std::make_unique<Node <T1, T2> ()>;
            }
        }
        size_t new_iT_in = iT_in;
        size_t new_iT_out = iT_out;
        if (new_iT_in > 0) {
            new_iT_in --;
            if (new_iT_out > 0) {
                new_iT_out --;
                // both T1 and T2 are not found
                return CreateNode<T2, Args...>::get(new_iT_in, new_iT_out);
            } else {
                // T2 is found
                if (!T2_is_first) {
                    // we just found T2 so we keep it in first place
                    return CreateNode<T2, Args...>::get(new_iT_in, new_iT_out, true, false);
                } else {
                    // T2 has been found at a previous call and is now T1, we keep it in first place
                    return CreateNode<T1, Args...>::get(new_iT_in, new_iT_out, true, false);
                }
            }
        } else {
            new_iT_out --;
            // T1 is found, we keep it in first place
            return CreateNode<T1, Args...>::get(new_iT_in, new_iT_out, false, false);
        }
    }
};

bool Eq_Float (float a, float b, float epsi) {
    return a < b + epsi && a > b - epsi;
}

float Random_Float (float a, float b, bool a_included = true, bool b_included = true) {
    return (
            ((float) rand() + (float) !a_included)
        ) / (
            (float) (RAND_MAX) + (float) !a_included + (float) !b_included
        ) * (b - a) + a;
}
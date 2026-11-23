#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace sandbox_visibility {

struct Bounds {
    std::array<double, 3> low{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> high{-low[0], -low[1], -low[2]};

    void include(const float* point) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(point[axis])) {
                low[axis] = high[axis] = std::numeric_limits<double>::quiet_NaN();
            } else {
                low[axis] = std::min(low[axis], double(point[axis]));
                high[axis] = std::max(high[axis], double(point[axis]));
            }
        }
    }

    bool finite() const {
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(low[axis]) || !std::isfinite(high[axis]) || low[axis] > high[axis])
                return false;
        return true;
    }
};

inline Bounds transformed(const Bounds& local, const float rows[4][3]) {
    if (!local.finite()) return {};
    Bounds world;
    for (int axis = 0; axis < 3; ++axis) {
        double low = rows[3][axis], high = low, magnitude = std::abs(low);
        for (int source = 0; source < 3; ++source) {
            const double a = local.low[source] * rows[source][axis];
            const double b = local.high[source] * rows[source][axis];
            low += std::min(a, b);
            high += std::max(a, b);
            magnitude += std::max(std::abs(a), std::abs(b));
        }
        // Enclose binary32 world transforms, including cancellation and fused arithmetic.
        const double error = 16 * std::numeric_limits<float>::epsilon() * std::max(1.0, magnitude);
        world.low[axis] = low - error;
        world.high[axis] = high + error;
    }
    return world;
}

class Frustum {
public:
    explicit Frustum(const float matrix[16]) {
        for (int i = 0; i < 16; ++i)
            valid_ = valid_ && std::isfinite(matrix[i]);
        // Row-vector D3D clip space: -w <= x,y <= w and 0 <= z <= w.
        const int columns[6] = {0, 0, 1, 1, 2, 2};
        const int signs[6] = {1, -1, 1, -1, 1, -1};
        for (int plane = 0; plane < 6; ++plane) {
            for (int row = 0; row < 4; ++row) {
                const double component = matrix[row * 4 + columns[plane]];
                const double w = plane == 4 ? 0 : matrix[row * 4 + 3];
                planes_[plane][row] = w + signs[plane] * component;
                magnitudes_[plane][row] = std::abs(w) + std::abs(component);
            }
        }
    }

    bool visible(const Bounds& bounds) const {
        if (!valid_ || !bounds.finite()) return true;
        for (int plane = 0; plane < 6; ++plane) {
            double distance = planes_[plane][3], magnitude = magnitudes_[plane][3];
            for (int axis = 0; axis < 3; ++axis) {
                const double coefficient = planes_[plane][axis];
                distance += coefficient * (coefficient >= 0 ? bounds.high[axis] : bounds.low[axis]);
                magnitude += magnitudes_[plane][axis] *
                             std::max(std::abs(bounds.low[axis]), std::abs(bounds.high[axis]));
            }
            // Cover both four-term clip dot products and their addition at a boundary.
            const double error = 8 * std::numeric_limits<float>::epsilon() * std::max(1.0, magnitude);
            if (distance < -error) return false;
        }
        return true;
    }

private:
    std::array<std::array<double, 4>, 6> planes_{}, magnitudes_{};
    bool valid_ = true;
};

}  // namespace sandbox_visibility

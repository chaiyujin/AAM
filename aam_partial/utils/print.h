#pragma once

#include <core/aam.h>

namespace aam {
	inline void print(const RowVectorX &x, int size = -1) {
		if (size < 0 || size > x.size()) size = x.size();
		printf("[");
		for (int i = 0; i < size; ++i) {
			if (x[i] < 0)
				printf("%.4f", x[i]);
			else
				printf("%.5f", x[i]);
			if (i < size - 1) printf(", ");
		}
		printf("]");
	}
}
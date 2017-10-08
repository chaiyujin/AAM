#pragma once

#include <core/aam.h>
#include <vector>
#include "triangulation.h"

namespace aam {
	class Mesh {
	private:
		BBox bbox;
		int lastHitIndex;
		std::vector<Triangle> triangleList;
	public:
		Mesh() { lastHitIndex = 0; }
		Mesh(const RowVectorX &shape);
		bool isInside(const Point &p, int &tri, double &alpha, double &beta, double &gamma);
		bool isInside(const Point &p) {
			int tri;
			double a, b, c;
			return isInside(p, tri, a, b, c);
		}
		BBox getBBox() const { return bbox; }
		void drawMesh(cv::Mat &image) const;
	};
}
#pragma once
#include <core/aam.h>
#include "bbox.h"
#include <opencv2/opencv.hpp>
#define MIN3(x, y, z) std::min(x, std::min(y, z))
#define MAX3(x, y, z) std::max(x, std::max(y, z))

namespace aam {
	class Triangle {
	private:
		Scalar x1, y1, x2, y2, x3, y3;
		Scalar dD;
		void calcdD();
	public:
		Triangle(int v1, int v2, int v3, const RowVectorX &allPoints);
		Triangle(int v1, int v2, int v3, int row, const MatrixX &allPoints);
		bool Triangle::isInside(const Point &p) const {
			Scalar a, b, c;
			return isInside(p, a, b, c);
		}
		bool Triangle::isInside(const Point &p, Scalar &alpha, Scalar &beta, Scalar &gamma) const;
		Point point(int i) const {
			assert((0 <= i && i <= 2));
			if (i == 0) return Point(x1, y1);
			if (i == 1) return Point(x2, y2);
			return Point(x3, y3);
		}
		BBox getBBox() const {
			Point p0(MIN3(x1, x2, x3), MIN3(y1, y2, y3));
			Point p1(MAX3(x1, x2, x3), MAX3(y1, y2, y3));
			return BBox(p0, p1);
		}
	};


	class Triangulation {
	private:
		static RowVectorXi m_triangleIndexes;
	public:
		friend class Saver;
		static RowVectorXi delaunayTriangulation(Eigen::Ref<const RowVectorX> ileavedPoints);
		static RowVectorXi getTriangleIndexes() {
			assert(m_triangleIndexes.rows() != 0);
			return m_triangleIndexes;
		}
	};
}
#include "triangulation.h"

namespace aam {
	RowVectorXi Triangulation::m_triangleIndexes;

	Triangle::Triangle(int v1, int v2, int v3, const RowVectorX &vec) {
		x1 = vec.col(v1 * 2).value();
		y1 = vec.col(v1 * 2 + 1).value();
		x2 = vec.col(v2 * 2).value();
		y2 = vec.col(v2 * 2 + 1).value();
		x3 = vec.col(v3 * 2).value();
		y3 = vec.col(v3 * 2 + 1).value();
		calcdD();
	}

	Triangle::Triangle(int v1, int v2, int v3, int row, const MatrixX &allPoints) {
		const auto &vec = allPoints.row(row);
		x1 = vec.col(v1 * 2).value();
		y1 = vec.col(v1 * 2 + 1).value();
		x2 = vec.col(v2 * 2).value();
		y2 = vec.col(v2 * 2 + 1).value();
		x3 = vec.col(v3 * 2).value();
		y3 = vec.col(v3 * 2 + 1).value();
		calcdD();
	}
	void Triangle::calcdD() {
		dD = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
	}

	bool Triangle::isInside(const Point &p,
		Scalar &alpha,
		Scalar &beta,
		Scalar &gamma) const {

		Scalar x, y;
		bool inSide;

		x = p.x;
		y = p.y;

		// perform bounding box test on x
		if (x<MIN3(x1, x2, x3) || x>MAX3(x1, x2, x3)) return false;

		// perform bounding box test on y
		if (y<MIN3(y1, y2, y3) || y>MAX3(y1, y2, y3)) return false;

		alpha = (y2 - y3) * (x - x3) + (x3 - x2) * (y - y3);
		beta = (y3 - y1) * (x - x3) + (x1 - x3) * (y - y3);
		gamma = dD - alpha - beta;

		inSide = alpha >= 0.0 && alpha <= dD &&
				 beta >= 0.0 && beta <= dD  &&
				 gamma >= 0.0 && gamma <= dD;

		if (inSide) {

			alpha /= dD;
			beta /= dD;
			gamma /= dD;
		}

		return inSide;
	}

	RowVectorXi Triangulation::delaunayTriangulation(Eigen::Ref<const RowVectorX> ileavedPoints) {
		auto points = toSeparatedViewConst<Scalar>(ileavedPoints);
		eigen_assert(points.cols() == 2);

		// Find min max.
		RowVector2 minC = points.colwise().minCoeff();
		RowVector2 maxC = points.colwise().maxCoeff();

		// Don't make the bounds too tight.
		cv::Rect_<Scalar> bounds(
			std::floor(minC.x() - aam::Scalar(1)),
			std::floor(minC.y() - aam::Scalar(1)),
			std::ceil(maxC.x() - minC.x() + aam::Scalar(2)),
			std::ceil(maxC.y() - minC.y() + aam::Scalar(2)));

		cv::Subdiv2D subdiv(bounds);

		std::vector<cv::Point2f> controlPoints;

		for (MatrixX::Index i = 0; i < points.rows(); ++i) {
			Point c(points(i, 0), points(i, 1));
			subdiv.insert(c);
			controlPoints.push_back(c);
		}

		std::vector<cv::Vec6f> triangleList;
		subdiv.getTriangleList(triangleList);

		RowVectorXi triangleIds(triangleList.size() * 3);

		int validTris = 0;
		for (size_t i = 0; i < triangleList.size(); i++)
		{
			cv::Vec6f t = triangleList[i];

			cv::Point2f p0(t[0], t[1]);
			cv::Point2f p1(t[2], t[3]);
			cv::Point2f p2(t[4], t[5]);

			if (bounds.contains(p0) && bounds.contains(p1) && bounds.contains(p2)) {

				auto iter0 = std::find(controlPoints.begin(), controlPoints.end(), p0);
				auto iter1 = std::find(controlPoints.begin(), controlPoints.end(), p1);
				auto iter2 = std::find(controlPoints.begin(), controlPoints.end(), p2);

				triangleIds(validTris * 3 + 0) = (int)std::distance(controlPoints.begin(), iter0);
				triangleIds(validTris * 3 + 1) = (int)std::distance(controlPoints.begin(), iter1);
				triangleIds(validTris * 3 + 2) = (int)std::distance(controlPoints.begin(), iter2);

				++validTris;
			}
		}

		m_triangleIndexes = triangleIds.leftCols(validTris * 3);
		return m_triangleIndexes;
	}
}
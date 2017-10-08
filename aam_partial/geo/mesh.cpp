#include "mesh.h"

namespace aam {
	Mesh::Mesh(const RowVectorX &shape) {
		lastHitIndex = 0;
		const auto &trIndexes = Triangulation::getTriangleIndexes();
		// construct triangles
		for (int i = 0; i < trIndexes.size(); i += 3) {
			int v0 = trIndexes[i], v1 = trIndexes[i + 1], v2 = trIndexes[i + 2];
			triangleList.emplace_back(v0, v1, v2, shape);
			// bbox
			bbox ^= triangleList[triangleList.size() - 1].getBBox();
		}
	}

	bool Mesh::isInside(const Point &p, int &tri, double &alpha, double &beta, double &gamma) {
		int trCount = (int)triangleList.size();
		double a, b, c;
		// check last hit index
		bool hit = triangleList[lastHitIndex].isInside(p, a, b, c);
		// try other
		if (!hit) {
			int newHitIndex = lastHitIndex;
			for (int k = (lastHitIndex + 1) % trCount; k != lastHitIndex; k = (k + 1) % trCount) {
				bool res = triangleList[k].isInside(p, a, b, c);
				if (res) {
					newHitIndex = k;
					break;
				}
			}
			if (newHitIndex != lastHitIndex) {
				hit = true;
				lastHitIndex = newHitIndex;
			}
		}
		if (hit) {
			tri = lastHitIndex;
			alpha = a; beta = b; gamma = c;
		}
		return hit;
	}

	void Mesh::drawMesh(cv::Mat &image) const {
		for (int tri = 0; tri < triangleList.size(); ++tri) {
			for (int i = 0; i < 3; ++i) {
				cv::line(image,
					triangleList[tri].point(i),
					triangleList[tri].point((i + 1) % 3),
					CV_RGB(0, 255, 0), 1);
			}
		}
	}
}
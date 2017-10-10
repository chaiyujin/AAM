#include "texture.h"
#include <geo/triangulation.h>
#include <geo/procrustes.h>
#include <geo/bbox.h>
#include <vector>

namespace aam {
	int Texture::g_h;
	int Texture::g_w;
	MatrixX Texture::g_texCoords;
	Mesh Texture::g_meanMesh;
	MatrixX Texture::texCoordsInMeanMesh(Mesh &meanMesh) {
		g_meanMesh = meanMesh;
		BBox bbox = meanMesh.getBBox();
		std::vector<std::vector<Scalar>> coordsList;
		for (int y = (int)bbox.startY(); y <= (int)bbox.endY(); ++y) {
			for (int x = (int)bbox.startX(); x <= (int)bbox.endX(); ++x) {
				int tri;
				float a, b, c;
				bool hit = meanMesh.isInside(Point((Scalar)x, (Scalar)y), tri, a, b, c);
				// hit!
				if (hit) {
					std::vector<Scalar> coords;
					coords.push_back((Scalar)tri);
					coords.push_back(a);
					coords.push_back(b);
					coords.push_back(c);
					coordsList.push_back(coords);
				}
			}
		}
		g_texCoords.resize(coordsList.size(), coordsList[0].size());
		for (int i = 0; i < g_texCoords.rows(); ++i) {
			g_texCoords.row(i) = Eigen::Matrix<Scalar, 4, 1>(coordsList[i][0], coordsList[i][1], coordsList[i][2], coordsList[i][3]);
		}
		return g_texCoords;
	}

	Color Texture::bilinear(const Point &p, const cv::Mat &image) {
		int x = (int)p.x;
		int y = (int)p.y;

		float rx = p.x - x;
		float ry = p.y - y;
		Color c00 = colorAt(image, x, y);
		Color c01 = colorAt(image, x, y + 1);
		Color c10 = colorAt(image, x + 1, y);
		Color c11 = colorAt(image, x + 1, y + 1);
		Color c0 = c00 * (1 - ry) + c01 * ry;
		Color c1 = c10 * (1 - ry) + c11 * ry;
		return c0 * (1 - rx) + c1 * rx;
	}

}
#pragma once

#include <core/aam.h>
#include <geo/mesh.h>
#include <geo/procrustes.h>

namespace aam {
	class Texture {
	private:
		static int g_w, g_h;
		static Mesh g_meanMesh;
		static MatrixX g_texCoords;
	public:
		friend class Saver;
		static cv::Point2i round(const Point &p) {
			return cv::Point2i((int)std::round(p.x), (int)std::round(p.y));
		}
		static Color colorAt(const cv::Mat &image, int x, int y) {
			const unsigned char *ptr = image.ptr<unsigned char>(y);
			return cv::Scalar(ptr[x * 3], ptr[x * 3 + 1], ptr[x * 3 + 2]);
		}
		static Color bilinear(const Point &p, const cv::Mat &image);
		static MatrixX texCoordsInMeanMesh(Mesh &meanMesh);
		static RowVectorX warpTexToMeanMesh(const RowVectorX &rawShape, const cv::Mat &rawImage);
		static MatrixX &getTexCoords() {
			assert(g_texCoords.rows() > 0);
			return g_texCoords;
		}
		static MatrixX collectTextures(const MatrixX &rawShapeList, const std::vector<std::string> &imgList);
		static MatrixX collectTextures(const MatrixX &rawShapeList, cv::VideoCapture &videoCapture);
		static void renderTexOnMesh(const RowVectorX &texture, Mesh &mesh, cv::Mat &image);
		static void getResolution(int &w, int &h) {
			w = g_w;
			h = g_h;
		}
	};
}

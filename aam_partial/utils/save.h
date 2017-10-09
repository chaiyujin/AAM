#pragma once

#include <core/aam.h>
#include <utils/asf.h>
#include <geo/procrustes.h>
#include <geo/triangulation.h>
#include <geo/mesh.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <texture/texture.h>
#include <utils/pca.h>
#include <utils/model.h>
#include <utils/texfit.h>
#include <utils/string.h>

namespace aam {
	class Saver {
	private:
		static std::string saveInternal() {
			std::string ret = "";
			ret += String::matToBytes<RowVectorX, Scalar>(Procrustes::m_meanShape);
			ret += String::matToBytes<RowVectorXi, int>(Triangulation::m_triangleIndexes);
			ret += String::matToBytes<MatrixX, Scalar>(Texture::g_texCoords);
			ret += String::numToBytes(Texture::g_w);
			ret += String::numToBytes(Texture::g_h);
			return ret;
		}

		static void loadInternal(std::istream &str) {
			String::matFromBytes<RowVectorX, Scalar>(str, Procrustes::m_meanShape);
			String::matFromBytes<RowVectorXi, int>(str, Triangulation::m_triangleIndexes);
			String::matFromBytes<MatrixX, Scalar>(str, Texture::g_texCoords);
			String::numFromBytes(str, Texture::g_w);
			String::numFromBytes(str, Texture::g_h);
			Texture::g_meanMesh = Mesh(Procrustes::m_meanShape);
		}

	public:
		static std::string save(const TexFitModel &m) {
			std::string ret = "";
			ret += saveInternal();
			ret += m.toString();
			return ret;
		}

		static bool load(std::istream stream, TexFitModel &m) {
			loadInternal(stream);
			m.fromistream(stream);
			return true;
		}

		static std::string save(std::string filePath, TexFitModel &m) {
			std::string ret = "";
			ret += saveInternal();
			ret += m.toString();
			std::ofstream fout(filePath, std::ofstream::binary);
			fout.write(ret.c_str(), ret.length());
			fout.close();
			printf("Save size: %d\n", ret.length());
			return ret;
		}

		static bool load(std::string filePath, TexFitModel &m) {
			std::ifstream fin(filePath, std::ifstream::binary);
			if (!fin.good()) return false;
			loadInternal(fin);
			m.fromistream(fin);
			fin.close();
			return true;
		}
	};
}
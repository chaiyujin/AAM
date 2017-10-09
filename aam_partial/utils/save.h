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
			ret += String::matToBytes(Procrustes::m_meanShape);
			ret += String::matToBytes(Triangulation::m_triangleIndexes);
			ret += String::matToBytes(Texture::g_texCoords);
			ret += String::numToBytes(Texture::g_w);
			ret += String::numToBytes(Texture::g_h);
			return ret;
		}

		static void loadInternal(std::stringstream &str) {
			String::matFromBytes(str, Procrustes::m_meanShape);
			String::matFromBytes(str, Triangulation::m_triangleIndexes);
			String::matFromBytes(str, Texture::g_texCoords);
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

		static bool load(std::stringstream stream, TexFitModel &m) {
			loadInternal(stream);
			m.fromStringStream(stream);
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
			fin.seekg(0, fin.end);
			long size = fin.tellg();
			fin.seekg(0);
			// allocate memory for file content
			std::string str(size, ' ');
			char* buffer = new char[size];
			fin.read(buffer, size);
			fin.close();
			str.assign(buffer, size);
			printf("Load size: %d\n", str.length());
			delete buffer;

			std::stringstream stream(str);
			loadInternal(stream);
			m.fromStringStream(stream);
			return true;
		}
	};
}
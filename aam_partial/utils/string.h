#pragma once

#include <core/aam.h>
#define kk 1
namespace aam {
	class String {
	public:
		/* for num */

		template <typename T>
		static std::string numToBytes(T x) {
			std::string ret = "";
			size_t size = sizeof(T);
			byte *bytes = (byte *)(&x);
			for (int i = 0; i < size; ++i) {
				ret += (char)bytes[i];
			}
			return ret;
		}

		template <typename T>
		static void numFromBytes(std::string bytesStr, T &x) {
			size_t size = sizeof(T);
			byte *bytes = new byte[size];
			memset(bytes, 0, size);
			for (int i = 0; i < size; ++i) {
				//printf("%x ", (byte)bytesStr[i]);
				bytes[i] = (byte)(bytesStr[i]);
			}
			memcpy(&x, bytes, size);
		}

		template <typename T>
		static void numFromBytes(std::stringstream &bytesStr, T &x) {
			size_t size = sizeof(T);
			byte *bytes = new byte[size];
			memset(bytes, 0, size);
			unsigned char c;
			for (int i = 0; i < size; ++i) {
				bytesStr >> std::noskipws >> c;
				//printf("%x ", c);
				bytes[i] = (byte)c;
			}
			memcpy(&x, bytes, size);
		}

		template <typename T>
		static T numFromBytes(std::string bytesStr) {
			T x;
			numFromBytes(bytesStr, x);
			return x;
		}
		
		template <typename T>
		static T numFromBytes(std::stringstream &bytesStream) {
			T x;
			numFromBytes(bytesStream, x);
			return x;
		}

		/* for mat */
		template <typename T>
		static std::string matToBytes(const T &mat) {
			std::string ret = "";
			int rows = mat.rows(), cols = mat.cols();
			ret += numToBytes<int>(rows);
			ret += numToBytes<int>(cols);
			for (int i = 0; i < mat.rows(); ++i) {
				for (int j = 0; j < mat.cols(); ++j) {
					ret += numToBytes<float>(mat(i, j));
				}
			}
			return ret;
		}

		template <typename T>
		static void matFromBytes(std::stringstream &str, T &mat) {
			int rows, cols;
			numFromBytes(str, rows);
			numFromBytes(str, cols);
			mat.resize(rows, cols);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					float v;
					numFromBytes<float>(str, v);
					mat(i, j) = v;
				}
			}
		}
	};
}
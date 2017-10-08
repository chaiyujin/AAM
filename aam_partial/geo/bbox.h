#pragma once
#include <core/aam.h>

namespace aam {
	class BBox {
	private:
		Point leftTop;
		Scalar m_width, m_height;
	public:
		BBox() : leftTop(1e10, 1e10) { m_width = -1e10; m_height = -1e10; }
		BBox(const Point &p0, const Point &p1) : leftTop(p0) {
			m_width = p1.x - p0.x;
			m_height = p1.y - p0.y;
		}
		Scalar startX() const { return leftTop.x; }
		Scalar startY() const { return leftTop.y; }
		Scalar endX() const { return leftTop.x + m_width; }
		Scalar endY() const { return leftTop.y + m_height; }
		Scalar width() const { return m_width; }
		Scalar height() const { return m_height; }

		const BBox operator^(const BBox &b) const {
			Point p0, p1;
			p0.x = std::min(leftTop.x, b.leftTop.x);
			p0.y = std::min(leftTop.y, b.leftTop.y);
			p1.x = std::max(leftTop.x + m_width, b.leftTop.x + b.m_width);
			p1.y = std::max(leftTop.y + m_height, b.leftTop.y + b.m_height);
			return BBox(p0, p1);
		}
		const BBox &operator^=(const BBox &b) {
			Point p0, p1;
			p0.x = std::min(leftTop.x, b.leftTop.x);
			p0.y = std::min(leftTop.y, b.leftTop.y);
			p1.x = std::max(leftTop.x + m_width, b.leftTop.x + b.m_width);
			p1.y = std::max(leftTop.y + m_height, b.leftTop.y + b.m_height);
			leftTop = p0;
			m_width = p1.x - p0.x;
			m_height = p1.y - p0.y;
			return *this;
		}
	};
}
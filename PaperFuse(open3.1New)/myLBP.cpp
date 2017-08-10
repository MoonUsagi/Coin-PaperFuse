#include "myLBP.h"
#include <climits>
#include <math.h>

#ifndef NDEBUG
#   include <iomanip>
#endif

namespace Descriptor {

	std::array<std::vector<bool>, myLBP::MAX_LENGTH / 8> myLBP::m_avbUniformMap = {};
	std::array<std::vector<cv::Point2i>, myLBP::NUMBER_OF_PATTERNS> myLBP::m_SamplingPoints = {};

	myLBP::myLBP(void) : myBlockDescriptorBase() {
		Init();
	}

	myLBP::myLBP(const cv::Mat& mImage, int Pattern, cv::Size2i BlockSize) :
		myBlockDescriptorBase(mImage, BlockSize) {
		Init();
		m_iFeatureType = Pattern;
		SetAttributes(Pattern);
	}

	myLBP::~myLBP(void) {}

	//做uniformLBP
	void myLBP::Init(void) {
		m_bIsUniform = true;
		m_iRadius = 1;
		m_iLength = 8;

		if (m_SamplingPoints.at(0).empty() == true) {
			SetSamplingPoints();
			for (unsigned int iLength = 8; iLength <= myLBP::MAX_LENGTH; iLength += 8) {

				auto iUniformMapIndex = iLength / 8 - 1;
				myLBP::m_avbUniformMap.at(iUniformMapIndex).resize(UINT64_C(1) << iLength);

				for (unsigned int j = 0; j < m_avbUniformMap.at(iUniformMapIndex).size(); ++j) {
					myLBP::m_avbUniformMap.at(iUniformMapIndex).at(j) = IsUniform(j, iLength);
				}
			}
		}
	}

	//LBP取點的位子
	void myLBP::SetSamplingPoints(void) {
		using cv::Point2i;
		
		std::array<Point2i, 8> Location8_1 = {
			Point2i(-1, -1), Point2i(0, -1), Point2i(+1, -1), Point2i(+1, +0),
			Point2i(+1, +1), Point2i(0, +1), Point2i(-1, +1), Point2i(-1, +0),
		};
		for (auto pt : Location8_1) {
			m_SamplingPoints.at(0).push_back(pt);
		}
		
		//橫向LBP---------------------------------------------------------------------------------
		/*
		std::array<Point2i, 8> Location8_1 = {
		Point2i(-1, 1), Point2i(0, 1), Point2i(+1, 1), Point2i(-1, +0),
		Point2i(1, 0), Point2i(-1, -1), Point2i(0, -1), Point2i(1, -1),
		};
		for (auto pt : Location8_1) {
		m_SamplingPoints.at(0).push_back(pt);
		}
		*/
		//----------------------------------------------------------------------------------------

		std::array<Point2i, 8> Location8_2 = {
			Point2i(-2, -2), Point2i(+0, -2), Point2i(+2, -2), Point2i(+2, +0),
			Point2i(+2, +2), Point2i(+0, +2), Point2i(-2, +2), Point2i(-2, +0)
		};
		for (auto pt : Location8_2) {
			m_SamplingPoints.at(1).push_back(pt);
		}

		std::array<Point2i, 16> Location16_2 = {
			Point2i(-2, -2), Point2i(-1, -2), Point2i(+0, -2), Point2i(+1, -2),
			Point2i(+2, -2), Point2i(+2, -1), Point2i(+2, +0), Point2i(+2, +1),
			Point2i(+2, +2), Point2i(+1, +2), Point2i(+0, +2), Point2i(-1, +2),
			Point2i(-2, +2), Point2i(-2, +1), Point2i(-2, +0), Point2i(-2, -1)
		};
		for (auto pt : Location16_2) {
			m_SamplingPoints.at(2).push_back(pt);
		}

		std::array<Point2i, 24> Location24_3 = {
			Point2i(-3, -3), Point2i(-2, -3), Point2i(-1, -3), Point2i(+0, -3),
			Point2i(+1, -3), Point2i(+2, -3), Point2i(+3, -3), Point2i(+3, -2),
			Point2i(+3, -1), Point2i(+3, +0), Point2i(+3, +1), Point2i(+3, +2),
			Point2i(+3, +3), Point2i(+2, +3), Point2i(+1, +3), Point2i(+0, +3),
			Point2i(-1, +3), Point2i(-2, +3), Point2i(-3, +3), Point2i(-3, +2),
			Point2i(-3, +1), Point2i(-3, +0), Point2i(-3, -1), Point2i(-3, -2)
		};
		for (auto pt : Location24_3) {
			m_SamplingPoints.at(3).push_back(pt);
		}
	}


	void myLBP::SetAttributes(int iPattern) {
		m_bIsUniform = ((iPattern & UNIFORM_FLAG) ? true : false);
		iPattern &= ~UNIFORM_FLAG;
		//------------------需修改成3個模式的----------------
		//半徑
		m_iRadius = ((iPattern == Feature::LBP_16_2) ? 2 : 1);
		//取點數量(長度)
		m_iLength = ((iPattern == Feature::LBP_16_2) ? 16 : 8);
	}


	//讀圖設定LBP票箱和取特徵計算
	void myLBP::Describe(cv::Point2i Position, std::vector<float>& viFeature) const {
		auto iUniformMapIndex = m_iLength / 8 - 1;
		std::vector<float> viTempBins(m_avbUniformMap.at(iUniformMapIndex).size(),
			0.0f);

		for (int y = Position.y; y < Position.y + m_BlockSize.height; ++y) {
			for (int x = Position.x; x < Position.x + m_BlockSize.width; ++x) {
				auto BinNumber = GetBinNumber(cv::Point2i(x, y));
				//計算票箱
				if (BinNumber != UINT_MAX) {
					++viTempBins.at(BinNumber);
				}
			}
		}

		viFeature.clear();
		//iNonuniformBin 非uniform票箱
		//viTempBins 一般票箱?  
		float iNonuniformBin = 0.0f;
		for (std::size_t i = 0; i < myLBP::m_avbUniformMap.at(iUniformMapIndex).size(); ++i) {
			//如果是Uniform的狀況
			if (m_bIsUniform == true) {
				if (myLBP::m_avbUniformMap.at(iUniformMapIndex).at(i) == true) {
					viFeature.push_back(viTempBins.at(i));
				}
				else {
					iNonuniformBin += viTempBins.at(i);
				}
			}
			else {
				viFeature.push_back(viTempBins.at(i));
			}
		}

		if (m_bIsUniform == true) {
			viFeature.push_back(iNonuniformBin);
		}
	}

	//計算位子
	unsigned int myLBP::GetBinNumber(cv::Point2i Position) const {
		//半徑
		cv::Point2i ptRadius(m_iRadius, m_iRadius);
		//左上(位子減半徑XY)
		cv::Point2i ptLeftTop(Position - ptRadius);
		//右下(位子減半徑XY)
		cv::Point2i ptRightBottom(Position + ptRadius + cv::Point2i(1, 1));

		//看有無超過邊界
		if (ptLeftTop.x < 0 || ptRightBottom.x >= m_mImage.cols ||
			ptLeftTop.y < 0 || ptRightBottom.y >= m_mImage.rows) {
			return UINT_MAX;
		}

		//沒有就進行取值
		return GetBinNumber(m_mImage(cv::Rect(ptLeftTop, ptRightBottom)));
	}

	//取值
	unsigned int myLBP::GetBinNumber(const cv::Mat& mImg) const {
		//
		unsigned int iBinNumber = 0;
		//中心點
		const cv::Point2i ptCenter(m_iRadius, m_iRadius);
		const size_t SamplingIdx =
			(m_iFeatureType & PATTERN_MASK & ~UNIFORM_FLAG) >> PATTERN_OFFSET;

		const auto& SampleingPoints = m_SamplingPoints.at(SamplingIdx);
		auto cCenterIntensity = mImg.at<unsigned char>(ptCenter);

		//平均值+中心點//------------------------------------------------------------
		
		/*
		    int iCount = 0;
			int nCenterIntensity = cCenterIntensity;
			for (const auto pt : SampleingPoints) 
			{
			nCenterIntensity += mImg.at<unsigned char>(ptCenter + pt);
			iCount++;
			}
			nCenterIntensity /= iCount + 1;

			//掃周遭點一個一個進去判斷為1或是0
			for (const auto pt : SampleingPoints) {

				auto cCurrentIntensity = mImg.at<unsigned char>(ptCenter + pt);
				//  cCurrentIntensity  周遭點 
				//  cCenterIntensity   中心點
				//  當周遭點小於中心點為0 反之為1
					iBinNumber = (iBinNumber << 1) |
						((cCurrentIntensity <= nCenterIntensity) ? 0x00 : 0x01);
			}

        */
		//-------------------------------------------------------------------


		//平均值//------------------------------------------------------------
		
		/*
		int iCount = 0;
		int nCenterIntensity = 0;
		for (const auto pt : SampleingPoints)
		{
			nCenterIntensity += mImg.at<unsigned char>(ptCenter + pt);
			iCount++;
		}
		nCenterIntensity /= iCount;

		//掃周遭點一個一個進去判斷為1或是0
		for (const auto pt : SampleingPoints) {

			auto cCurrentIntensity = mImg.at<unsigned char>(ptCenter + pt);
			//  cCurrentIntensity  周遭點 
			//  cCenterIntensity   中心點
			//  當周遭點小於中心點為0 反之為1
			iBinNumber = (iBinNumber << 1) |
				((cCurrentIntensity <= nCenterIntensity) ? 0x00 : 0x01);
		}
		
		*/
		//--------------------------------------------------------------------


		
		//原始作法//----------------------------------------------------------
		
		//掃周遭點一個一個進去判斷為1或是0
		
		for (const auto pt : SampleingPoints) {

			auto cCurrentIntensity = mImg.at<unsigned char>(ptCenter + pt);
			//  cCurrentIntensity  周遭點 
			//  cCenterIntensity   中心點
			//  當周遭點小於中心點為0 反之為1
		
			iBinNumber = (iBinNumber << 1) |
				(((cCurrentIntensity) <= cCenterIntensity ) ? 0x00 : 0x01);
		}
		
		//--------------------------------------------------------------------
		


		//LTP//----------------------------------------------------------
		/*
		
		//掃周遭點一個一個進去判斷為1或是0


		for (const auto pt : SampleingPoints) {

			auto cCurrentIntensity = mImg.at<unsigned char>(ptCenter + pt);
			//  cCurrentIntensity  周遭點 
			//  cCenterIntensity   中心點
			//  當周遭點小於中心點為0 反之為1

			//iBinNumber = (iBinNumber << 1) |
			//(((cCurrentIntensity) >= cCenterIntensity -) ? 0x01 : 0x00);

			int DniBinNumber = (iBinNumber << 1) |
				(((cCurrentIntensity) <= cCenterIntensity + 5) ? 0x00 : 0x01);

			int UpiBinNumber = (iBinNumber << 1) |
				(((cCurrentIntensity) >= cCenterIntensity - 5) ? 0x01 : 0x00);

			iBinNumber = DniBinNumber;
			//iBinNumber = (DniBinNumber + UpiBinNumber)/2;
		}

		*/
		//--------------------------------------------------------------------



		//ELTP//----------------------------------------------------------
		/*
		//掃周遭點一個一個進去判斷為1或是0
         
           int iCount = 0;
		   int SDcCurrentIntensity = 0;
           for (const auto pt : SampleingPoints)
           {
			   SDcCurrentIntensity += mImg.at<unsigned char>(ptCenter + pt);
				iCount++;
			}
		      SDcCurrentIntensity /= iCount + 1;

		  // int SD = 0;
		  // for (const auto pt : SampleingPoints)
		  // {
			 //  SD += (mImg.at<unsigned char>(ptCenter + pt) - SDcCurrentIntensity)*(mImg.at<unsigned char>(ptCenter + pt) - SDcCurrentIntensity);
			 //  iCount++;
		  // }
		  //   SD /= iCount + 1;
			 //sqrt(SD);
		 
		//掃周遭點一個一個進去判斷為1或是0
				
		for (const auto pt : SampleingPoints) {

			auto cCurrentIntensity = mImg.at<unsigned char>(ptCenter + pt);

			//  cCurrentIntensity  周遭點 
			//  cCenterIntensity   中心點
			//  當周遭點小於中心點為0 反之為1
			
			SDcCurrentIntensity = (cCenterIntensity - SDcCurrentIntensity);
			pow(SDcCurrentIntensity,2);
			sqrt(SDcCurrentIntensity);


			int SD = SDcCurrentIntensity;  //標準差


			int DniBinNumber = (iBinNumber << 1) |
				(((cCurrentIntensity) <= cCenterIntensity + (SD*0.5)) ? 0x00 : 0x01);

			int UpiBinNumber = (iBinNumber << 1) |
				(((cCurrentIntensity) >= cCenterIntensity - (SD*0.5)) ? 0x01 : 0x00);

			//iBinNumber = UpiBinNumber;
			iBinNumber = (DniBinNumber + UpiBinNumber)/2;
		}
		*/
		//--------------------------------------------------------------------



		//-----抗旋轉--------------------------------------------------------- 
		/*
		//掃周遭點一個一個進去判斷為1或是0

		for (const auto pt : SampleingPoints) 
		{

		auto cCurrentIntensity = mImg.at<unsigned char>(ptCenter + pt);
			//  cCurrentIntensity  周遭點 
			//  cCenterIntensity   中心點
			//  當周遭點小於中心點為0 反之為1

			iBinNumber = (iBinNumber << 1) |
				(((cCurrentIntensity) <= cCenterIntensity) ? 0x00 : 0x01);
		}

	    unsigned int iMaskBit = 0x00000001;
		int iMin = 9999999;
		for (int iShift = 0; iShift < 8; iShift++)
		{
			int iShifted = iBinNumber &iMaskBit;
			iBinNumber = iBinNumber >> 1;
			iBinNumber = iBinNumber | (iShifted << 7);

			if (iBinNumber < iMin)
			{
				iMin = iBinNumber;
			}
		}
		//std::cout << iMin;
		return iMin;
		
		
	    */
		//-----抗旋轉-------------------------------------------------------------
		return iBinNumber;
	}

	bool myLBP::IsUniform(int iPatten, unsigned int iBinNumber) {
		auto iIndex = ((iPatten & ~UNIFORM_FLAG) == Feature::LBP_16_2) ? 1 : 0;
		return m_avbUniformMap.at(iIndex).at(iBinNumber);
	}

	//檢查uniform 
	//iBinNumber要檢查的數    iLength bit長度
	bool myLBP::IsUniform(unsigned int iBinNumber, unsigned int iLength) {
		bool bResult = true;
		//改變次數
		unsigned int iChangeTime = 0;

		unsigned int iMaskBit = 0x00000001;
		unsigned int iCheckBit = iBinNumber & iMaskBit;
		//計算跳動次數
		for (unsigned int i = 0; i < iLength - 1; ++i) {
			iMaskBit <<= 1;
			iCheckBit <<= 1;
			unsigned int iComparedValue = (iMaskBit & iBinNumber);
			//不等於時跳動次數+1
			if (iComparedValue != iCheckBit) {
				iCheckBit = iComparedValue;
				++iChangeTime;
				//如果跳動次數 >設定的極限(2次) bResult為false
				if (iChangeTime > myLBP::MAX_TRANSITION_TIME) {
					bResult = false;
					break;
				}
			}
		}

		return bResult;
	}

#ifndef NDEBUG
	void myLBP::PrintUniformMap(int iLength) const {
		int i = 0;
		int sum = 0;
		for (auto bIsUnoform : m_avbUniformMap.at(iLength / 8 - 1)) {
			std::cout << std::setw(3) << i++ << ":" << (bIsUnoform ? "T" : "F") << std::endl;
			if (bIsUnoform == true) {
				++sum;
			}
		}
		std::cout << sum << std::endl;
	}
#endif

};
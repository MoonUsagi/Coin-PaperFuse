#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "myImageSequence.h"
#include "common.h"
#include "myBlockDescriptor.h" 
#include "myScanner.h"
#include "mySVM.h"
//#include "myHOG.h"
#include <fstream>
#include <sstream>
#include <iomanip>

//Test
#define TrainQuantity 110           // Ans ��Ʈw   �V�m�˪O���
#define TestQuantity 55              // Train ��Ʈw ���Ѹ�� 
//#define TrainQuantity 26           // Ans ��Ʈw   �V�m�˪O���
//#define TestQuantity 13              // Train ��Ʈw ���Ѹ�� 
//#define FQuantity 65536
#define FQuantity 15104
//#define FQuantity 1764

// Normalize
const cv::Size2i GISTnormalize_size = cv::Size2i(512, 512);
const cv::Size2i normalize_size = cv::Size2i(256, 256);  //���W��
const cv::Size2i Lbpnormalize_size = cv::Size2i(128, 128);

//GIST 
bool GistTrain = false;
bool GistTest = false;


//Histogram    //����Ϥ��
bool bHistogramTrain = false;          //�ڦ��Z��
bool bHistogramTest = false;           //�ڦ��Z��  
bool bHistogramTwoTestTrain = false;   //�d�g�h�B �P�U�ؤ�k  �O�o�勵�W�Ƥj�p
cv::Mat TrainDataMat(TrainQuantity, FQuantity, CV_32FC1);
cv::Mat TestDataMat(TestQuantity, FQuantity, CV_32FC1);
char Path[100];
float counter = 0.0;
double HistMatch[TrainQuantity] = {};


// Hog attributes
const bool bUsingHOG = false;
const cv::Size block_size(64, 64);
const cv::Size stride_size(32, 32);
const cv::Size cell_size(32, 32);
const int bins = 9;

// LBP attributes
const bool bUsingLBP = true;
const cv::Size2i LBPBlock = cv::Point2i(8, 8);
const cv::Point2i LBPStep = cv::Point2i(LBPBlock.width, LBPBlock.height);
Descriptor::myBlockDescriptor oDenseDetector(cv::Mat(), cv::Size2i(16, 16));
const int Boundary = 128;

// Run  
const bool bTraining = true;
const bool bTesting = true;
const bool CambTesting = false;    //�۾�����
const std::string sModelName = "SVM.xml";
//const std::string sTrainingPath = "Train_GIST/";
//const std::string sTestingPath = "Test_GIST/";
//const std::string sTrainingPath = "CoinTrain/"; 
//const std::string sTestingPath = "CoinTest/";
//const std::string sTrainingPath = "NewTrain/";
//const std::string sTestingPath = "NewTest/";
const std::string sTrainingPath = "Train_Change_2/";
const std::string sTestingPath = "Test_Change_2/";
//const std::string sTrainingPath = "UTraining/";
//const std::string sTestingPath = "UTest/";
//const std::string sTrainingPath = "Train_Change_ALL/";
//const std::string sTestingPath = "Test_Change_ALL/";
//const std::string sTrainingPath = "trainttt/";
//const std::string sTestingPath = "RotateTestMs/";
//const std::string sTrainingPath = "SourceSmallTrain/";
//const std::string sTestingPath = "SourceSmallTest/";
//const std::string sTrainingPath = "RotateTrainM/";
//const std::string sTestingPath = "RotateTestMs/";
//const std::string sTrainingPath = "Train_S768/";
//const std::string sTestingPath = "Test_S768/";
//const std::string sTrainingPath = "Train_S512/";
//const std::string sTestingPath = "Test_S512/";
//const std::string sTrainingPath = "Train_Change_DWT/";
//const std::string sTestingPath = "Test_Change_DWT/";
//const std::string sTrainingPath = "Train_DWT/";
//const std::string sTestingPath = "Test_DWT/";
//const std::string sTrainingPath = "Train_DWT_S192/";
//const std::string sTestingPath = "Test_DWT_S192/";


std::vector<cv::Rect> vrDense;


//void SymmetyrHOG(cv::Mat clsImage, std::vector<float>& iBins)
//{
//	myHOG clsHogR = myHOG(clsImage, myHOG::Feature::HOG_WITHOUT_NORM, cv::Size(16, 16));
//
//	for (size_t iRow = 0; iRow < clsImage.rows - 8; iRow += 8)
//	{
//		for (size_t iCol = 0; iCol < clsImage.cols - 8; iCol += 8)
//		{
//			std::vector<float> vFeatureR;
//			clsHogR.Describe(cv::Point2i(iCol, iRow), vFeatureR);
//			float fSum = 0;
//
//			//Sysmetry
//			for (size_t i = 0; i < vFeatureR.size() - 18; i += 18)
//			{
//				for (size_t j = 0; j < 9; j++)
//				{
//					float fP = vFeatureR[i + j] - vFeatureR[i + j + 9];
//					if (fP < 0)
//						fP = (-fP);
//					fSum += fP;
//				}
//			}
//
//
//			for (size_t i = 0; i < vFeatureR.size() - 18; i++)
//			{
//				float fP = vFeatureR[i] - vFeatureR[i + 18];
//				if (fP < 0)fP = (-fP);
//				fSum += fP;
//			}
//
//			fSum = ((100 * fSum) / 9180);
//
//			iBins.push_back(fSum);
//
//		}
//	}
//}
//
//
//void HOG(cv::Mat clsImage, std::vector<float>& vFeatureR)
//{
//	myHOG clsHogR = myHOG(clsImage, myHOG::Feature::HOG_WITHOUT_NORM, cv::Size(16, 16));
//
//	for (size_t iRow = 0; iRow < clsImage.rows - 8; iRow += 8)
//	{
//		for (size_t iCol = 0; iCol < clsImage.cols - 8; iCol += 8)
//		{
//			clsHogR.Describe(cv::Point2i(iCol, iRow), vFeatureR);
//		}
//	}
//}

void CalFeature(cv::Mat& mImg, std::vector<float>& vfFeature)
{
	if (bUsingHOG)
	{
		cv::Size win_size = mImg.size();
		CvSize winshiftsize = cvSize(1, 1);
		CvSize paddingsize = cvSize(0, 0);

		cv::HOGDescriptor hog = cv::HOGDescriptor(win_size, block_size, stride_size, cell_size, bins);
		hog.compute(mImg, vfFeature, winshiftsize, paddingsize);
		std::cout << "����" << vfFeature.size();
	}

	if (bUsingLBP)  
	{
		cv::resize(mImg, mImg, Lbpnormalize_size);
		oDenseDetector.SetImage(mImg);
		std::vector<float> vfDenseLBP;
		for (const auto& r : vrDense)
		{
			//printf("%d\n", vrDense.size());
			std::vector<float> vfTemp;
			oDenseDetector.Describe(r, vfTemp);

			for (auto f : vfTemp)
			{
				vfDenseLBP.push_back(f);
			}
		}

		//std::cout << "����" << vfDenseLBP.size();

		vfFeature.reserve(vfFeature.size() + vfDenseLBP.size());
		for (const auto f : vfDenseLBP)
		{
			vfFeature.push_back(f);
		}
		//std::cout << "����" << vfFeature.size();
	}

}

void sort(float arr[], int len)               // �w�j�Ƨ�
{
	int i, j, temp;
	for (i = 0; i < len - 1; i++)
	{
		for (j = 0; j < len - 1 - i; j++)
		{
			if (arr[j] > arr[j + 1])
			{
				temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}
}

int main(void)
{
	Classifier::mySVM svm;

	if (!bTraining)
	{
		svm.Load(sModelName);
	}

	// claculate the rect for dense extractor
	{
		Plugin::myScanner scanner(cv::Point2i(0, 0), cv::Point2i(Boundary, Boundary));
		scanner.CalRect(vrDense, LBPBlock, LBPStep);
	}


	{
		const std::vector<int> viDenseFeature =
		{
			Descriptor::myBlockDescriptor::Feature::LBP_8_1_UNIFORM
		};
		// setting dense detector
		for (auto feature : viDenseFeature)
		{
			oDenseDetector.EnableFeature(feature);
		}
	}

	std::cout << "HOG_block_size" << block_size << std::endl
		<< "HOG_stride_size" << stride_size << std::endl
		<< "HOG_cell_size" << cell_size << std::endl << std::endl;

	std::cout << "LBP_block_size" << LBPBlock << std::endl
		<< "LBP_cell_size" << LBPStep << std::endl << std::endl;



	//bHistogramTrain-------------------------------------------------------------------------------------------------

	if (bHistogramTrain)
	{
		for (int i = 0; i < TrainQuantity; i++)
		{
			//sprintf_s(Path, sizeof(Path), "Train_Change/%03d.png", i);
			sprintf_s(Path, sizeof(Path), "CoinTrain/%03d.png", i);
			cv::Mat mImg = cv::imread(Path, cv::IMREAD_GRAYSCALE);
			if (mImg.empty())
			{
				std::cout << i << "Not Found!" << std::endl;
				exit(EXIT_FAILURE);
			}

			cv::resize(mImg, mImg, Lbpnormalize_size);
			//cv::resize(mImg, mImg, normalize_size);

			std::vector<float> vfHLFeature;
			CalFeature(mImg, vfHLFeature);

			for (int j = 0; j < vfHLFeature.size(); j++)
			{
				TrainDataMat.at<float>(i, j) = vfHLFeature[j];
			}
		}
	}

	if (bHistogramTest)
	{

		for (int j = 0; j < TestQuantity; j++)
		{
			//sprintf_s(Path, sizeof(Path), "Test_Change/%03d.png", j);
			sprintf_s(Path, sizeof(Path), "CoinTest/%03d.png", j);
			cv::Mat mImg = cv::imread(Path, cv::IMREAD_GRAYSCALE);
			if (mImg.empty())
			{
				std::cout << j << "Not Found!" << std::endl;
				exit(EXIT_FAILURE);
			}

			//���W�� ���S�x��
			cv::resize(mImg, mImg, Lbpnormalize_size);
			//cv::resize(mImg, mImg, normalize_size);
			
			std::vector<float> vfHLFeature;
			CalFeature(mImg, vfHLFeature);

			//�ܼƩI�s
			std::vector<int> Ans;
			int Anss = 0;
			float distances[TrainQuantity] = {};
			float SortDistances[TrainQuantity] = {};
			float MinAns = 0;

			//���ڦ��Z��
			for (int f = 0; f < TrainQuantity; f++)
			{
				float distance = 0;

				for (int k = 0; k < vfHLFeature.size(); k++)
				{
					distance += abs(TrainDataMat.at<float>(f, k) - vfHLFeature[k]);
				}

				distances[f] = distance;                 // �N�ڦ��Z���۴�᪺��ƥ�J�}�C
			}


			for (int i = 0; i < sizeof(SortDistances) / sizeof(SortDistances[0]); i++)
			{
				SortDistances[i] = distances[i];
			}

			sort(SortDistances, TestQuantity);  // �}�C�Ƨ�
			MinAns = SortDistances[0];    // ���o�}�C�̤p��   

			for (int index = 0; index < TestQuantity+1; index++)
			{
				if (distances[index] == MinAns)
				{
					Anss = index;
					Ans.push_back(index);
				}
			}

			std::cout << std::endl;
			std::cout << "Anss(���̨ε��G) : " << Anss / 2 << std::endl;

			if (j == (Anss / 2))
			{
				std::cout << j << "���T" << std::endl;
				counter++;
			}
			else
			{
				std::cout << j << "�����T" << std::endl;
			}

		}
		float result = (counter / 55);
		std::cout << std::endl << "���T�ƶq:" << counter << std::endl << "���Ѳv:" << result;
	}
	
	//-----------------------------------------------------------------------------------------------------------------


	//bHistogramTwoTestTrain-------------------------------------------------------------------------------------------
	if (bHistogramTwoTestTrain)
	{
		for (int i = 0; i < TestQuantity; i++)
		{
			//���

			//sprintf_s(Path, sizeof(Path), "Test_Change/%03d.png", i);
			sprintf_s(Path, sizeof(Path), "CoinTest/%03d.png", i);

			//��X����
			//std::cout << "���ռv��  " << i << std::endl;
			//printf("%s\n", Path);

			//Ū��
			cv::Mat mImg = cv::imread(Path, cv::IMREAD_GRAYSCALE);

			// ����䤣���
			if (mImg.empty())
			{
				std::cout << i << "Not Found!" << std::endl;
				exit(EXIT_FAILURE);
			}

			//cv::resize(mImg, mImg, Lbpnormalize_size);
			cv::resize(mImg, mImg, normalize_size);

			std::vector<float> vfHLFeature;
			CalFeature(mImg, vfHLFeature);

			cv::Mat TestTwoDataMat(1, FQuantity, CV_32F);

			for (int j = 0; j < vfHLFeature.size(); j++)
			{
				TestTwoDataMat.at<float>(0, j) = vfHLFeature[j];
			}


			// �ĤG���V�m�˥���J
			for (int j = 0; j < TrainQuantity; j++)
			{
				//���

				//sprintf_s(Path, sizeof(Path), "Train_Change/%03d.png", j);
				sprintf_s(Path, sizeof(Path), "CoinTrain/%03d.png", j);

				//��X����
				//std::cout << std::endl << "�V�m�v��" << j << std::endl;
				//printf("%s\n", Path);

				//Ū��
				cv::Mat TImg = cv::imread(Path, cv::IMREAD_GRAYSCALE);

				// ����䤣���
				if (TImg.empty())
				{
					std::cout << "Not Found!" << std::endl;
					break;
				}

				//cv::resize(TImg, TImg, Lbpnormalize_size);
				cv::resize(TImg, TImg, normalize_size);

				std::vector<float> vTrainfHLFeature;
				CalFeature(TImg, vTrainfHLFeature);

				cv::Mat TrainTwoDataMat(1, FQuantity, CV_32F);

				for (int j = 0; j < vTrainfHLFeature.size(); j++)
				{
					TrainTwoDataMat.at<float>(0, j) = vTrainfHLFeature[j];
				}
				HistMatch[j] = cv::compareHist(TrainTwoDataMat, TestTwoDataMat, CV_COMP_INTERSECT);
				//HistMatch[j] = cv::EMD(TestTwoDataMat, TrainTwoDataMat,CV_DIST_L2);
			}

			//�ܼƩI�s
			std::vector<int> Ans;
			int Anss = 0;
			std::vector<double> SortDistances(TrainQuantity);
			//float SortDistances[TrainQuantity] = {};
			double MinAns = 0;

			for (int i = 0; i < TrainQuantity; i++)
			{
				SortDistances[i] = HistMatch[i];
			}

			cv::sort(SortDistances, SortDistances, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

			MinAns = SortDistances[0];    // ���o�}�C�̤p��   

			for (int index = 0; index < TrainQuantity; index++)
			{
				if (HistMatch[index] == MinAns)
				{
					Anss = index;
					Ans.push_back(index);
				}
			}

			for (size_t i = 0; i < Ans.size(); ++i) {
				std::cout << Ans[i] << " ";
			}

			std::cout << std::endl;

			std::cout << "Anss(���̨ε��G) : " << Anss / 2 << std::endl;

			if (i == (Anss / 2))
			{
				std::cout << i << "���T" << std::endl;
				counter++;
			}
			else
			{
				std::cout << i << "�����T" << std::endl;
			}

		}
		float result = (counter / 55);
		std::cout << std::endl << "���T�ƶq:" << counter << std::endl << "���Ѳv:" << result;

	}

	//-----------------------------------------------------------------------------------------------------------------
	if (bTraining)
	{

		//myImageSequence seq(sTrainingPath, "", "jpg", false);
		myImageSequence seq(sTrainingPath, "", "png", false);
		//myImageSequence seq("Training/", "", "png", false);
		seq.SetAttribute(myImageSequence::Attribute::PADDING_LENGTH, 3);
		
		cv::Mat mImg;
		 
		while (seq >> mImg)
		{
			cv::resize(mImg, mImg, normalize_size);

			std::vector<float> vfHLFeature;
			CalFeature(mImg, vfHLFeature);

			svm.AddSample(seq.GetSequenceNumber()/2, vfHLFeature);
		}

		svm.Train();
		svm.Save(sModelName);
	}

	if (bTesting)
	{
		std::vector<float> vfAns;
		//myImageSequence seq(sTestingPath, "", "jpg", false);
		myImageSequence seq(sTestingPath, "", "png", false);
		//myImageSequence seq("Training/", "", "png", false);
		seq.SetAttribute(myImageSequence::Attribute::PADDING_LENGTH, 3);

		cv::Mat mImg;

		while (seq >> mImg)
		{
			cv::resize(mImg, mImg, normalize_size);

			std::vector<float> vfHLFeature;
			CalFeature(mImg, vfHLFeature);

			float fResult = svm.Predict(vfHLFeature);
			vfAns.push_back(fResult);

			std::cout  << fResult << std::endl;
			// << "output: "

		}

		int iSum = 0;
		for (size_t i = 0; i < vfAns.size(); i++)
		{
			if (vfAns.at(i) == i)
			{
				std::cout << "�諸���O: " << i << std::endl;
				iSum++;
			}
		}

		std::cout << "Total: " << vfAns.size() << std::endl;
		std::cout << "Corect Count: " << iSum << std::endl;
		std::cout << "Ratio: " << static_cast<float>(iSum) / vfAns.size() * 100 << "%" << std::endl;

	}

	//--------------------------------------------------------------------------------------------------------------

	//cam
	if (CambTesting)
	{
		cv::VideoCapture cap(0);
		while (true)
		{
			cv::Mat Img;
			cap >> Img;
			if (cv::waitKey(1) >= 0)
			{
				if (Img.empty())
				{
					std::cout << "Not Found!" << std::endl;
					exit(EXIT_FAILURE);
				}
				else
				{
					std::vector<float> vfAns;

					cv::imshow("pCam", Img);
					cv::resize(Img, Img, normalize_size);

					std::vector<float> vfHLFeature;
					CalFeature(Img, vfHLFeature);

					float fResult = svm.Predict(vfHLFeature);
					vfAns.push_back(fResult);

					std::cout << "output: " << fResult << std::endl;

					int FinalResult = fResult;
					switch (FinalResult)
					{
					  case 0:
						  std::cout << "����@��";
						  break;
					  case 1:
						  std::cout << "�������";
						  break;
					  case 2:
						  std::cout << "����Q��";
						  break;
					  case 3:
						  std::cout << "����@�ʤ�";
						  break;
					  case 4:
						  std::cout << "�x���@��";
						  break;
					  case 5:
						  std::cout << "�x������";
						  break;
					  case 6:
						  std::cout << "�x���Q��";
						  break;
					  case 7:
						  std::cout << "�x�����Q��";
						  break;
					  case 8:
						  std::cout << "�s������";
						  break;
					  case 9:
						  std::cout << "�s���Q��";
						  break;
					  case 10:
						  std::cout << "�s���G�Q��";
						  break;
					  case 11:
						  std::cout << "�s�����Q��";
						  break;
					  case 12:
						  std::cout << "�s���@��";
						  break;
					}
				}
			}
			

			cv::imshow("Cam", Img);
			cv::waitKey(1);
		}
	}


	//GIST----------------------------------------------------------------------------------------------------------
	
	if (GistTrain)
	{
		for (int i = 0; i < TestQuantity; i++)
		{
			sprintf_s(Path, sizeof(Path), "Test_GIST/%03d.jpg", i);

			cv::Mat mImg = cv::imread(Path, cv::IMREAD_COLOR);

			if (mImg.empty())
			{
				std::cout << i << "Not Found!" << std::endl;
				exit(EXIT_FAILURE);
			}
			cv::imshow("mm", mImg);
			cv::resize(mImg, mImg, GISTnormalize_size);
			
			// �ĤG���V�m�˥���J
			for (int j = 0; j < TrainQuantity; j++)
			{
				sprintf_s(Path, sizeof(Path), "Train_GIST/%03d.jpg", j);

				cv::Mat TImg = cv::imread(Path, cv::IMREAD_COLOR);

				if (TImg.empty())
				{
					std::cout << "Not Found!" << std::endl;
					break;
				}

				cv::resize(TImg, TImg, GISTnormalize_size);

				//HistMatch[j] = cv::compareHist(TImg, mImg, CV_COMP_CHISQR_ALT);
				//HistMatch[j] = cv::EMD(TImg, mImg, CV_DIST_L2);
			}
			 
			//�ܼƩI�s
			std::vector<int> Ans;
			int Anss = 0;
			std::vector<double> SortDistances(TrainQuantity);
			//float SortDistances[TrainQuantity] = {};
			double MinAns = 0;

			for (int i = 0; i < TrainQuantity; i++)
			{
				SortDistances[i] = HistMatch[i];
			}

			cv::sort(SortDistances, SortDistances, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

			MinAns = SortDistances[0];    // ���o�}�C�̤p��   

			for (int index = 0; index < TrainQuantity; index++)
			{
				if (HistMatch[index] == MinAns)
				{
					Anss = index;
					Ans.push_back(index);
				}
			}

			for (size_t i = 0; i < Ans.size(); ++i) {
				std::cout << Ans[i] << " ";
			}

			std::cout << std::endl;

			std::cout << "Anss(���̨ε��G) : " << Anss / 2 << std::endl;

			if (i == (Anss / 2))
			{
				std::cout << i << "���T" << std::endl;
				counter++;
			}
			else
			{
				std::cout << i << "�����T" << std::endl;
			}

		}
		float result = (counter / 55);
		std::cout << std::endl << "���T�ƶq:" << counter << std::endl << "���Ѳv:" << result;

	}



	system("pause");
	return 0;
}

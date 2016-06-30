#ifndef MULTIF0FEATURE
#define MULTIF0FEATURE

#include "featureExtractor/featureExtractor.h"
#include "general.h"
#include <aubio.h>

class MultiF0FeatureExtractor : public FeatureExtractor
{
public:

    MultiF0FeatureExtractor( int formant, aubio_pvoc_t*  pvIn )
    {
        selectedFormant = formant;
        char cStr[] = "default";
        pv = pvIn;
        samples = cv::Mat(  1, 1 , CV_64FC1 );
        pitchOut = new_fvec (1); // output candidate
        pitch = new_aubio_pitch (cStr, win_s, hopSize, sampleRate);
    }

    ~MultiF0FeatureExtractor()
    {
        del_aubio_pitch (pitch);
        del_fvec (pitchOut);
    }

    void getFormants( double f0, cvec_t* inputComplex )
    {
       double freqStep = sampleRate / win_s;
       std::array< std::pair<double, double> , 5> formants;
       formants[0].first = f0;
       for ( int curFreq = f0; curFreq < 5000; curFreq += f0 )
       {
            if ( curFreq < 1000 )
                continue;

            int formant = curFreq / 1000;
            int formantIndex = curFreq / freqStep;
            float curFormantVal = inputComplex->norm[formantIndex];
            if ( formants[formant].second < curFormantVal )
            {
                formants[formant].first = curFreq % 1000;
                formants[formant].second = curFormantVal;
            }
       }

       if ( selectedFormant != -1 )
       {
           if ( selectedFormant == 0 )
            samples.at<double>(colSize, 0) = (formants[selectedFormant].first - MIN_FREQ) / JUMPSIZE;
           else
            samples.at<double>(colSize, 0) = (formants[selectedFormant].first) / 25 ;// / (JUMPSIZE * 5);
       }
    }

    virtual DataType2D& getFeatures() override
    {
        return samples;
    }

    void getSecondPitch( double f0, cvec_t* inputComplex, fvec_t *inputSimple )
     {
         int offSet = 2;
         double freqStep = sampleRate / win_s;

         for ( size_t curFreq = f0; curFreq < (sampleRate/2 - curFreq) ; curFreq += f0 )
         {
             int formantIndex = curFreq / freqStep;
             for ( int curIndex = formantIndex - offSet; curIndex < formantIndex + offSet; curIndex++)
             {
                 inputComplex->norm[curIndex] = 0;
                 inputComplex->phas[curIndex] = 0;
             }
         }
         aubio_pvoc_rdo( pv, inputComplex, inputSimple);
         aubio_pitch_do (pitch, inputSimple, pitchOut);
         if ( pitchOut->data[0] < MIN_FREQ || pitchOut->data[0] > MAX_FREQ )
             return;
         else
             std::cout << pitchOut->data[0];

     }

    void swapInputWithIn( cvec_t *inputComplex )
    {
        auto middlePos = inputComplex->norm + inputComplex->length/2;
        std::vector<double> returnVal( middlePos , inputComplex->norm + inputComplex->length );
        returnVal.insert( returnVal.end(), inputComplex->norm, middlePos);
        for ( size_t i = 0; i < inputComplex->length; i++ )
            inputComplex->norm[i] = returnVal[i];

        middlePos = inputComplex->phas + inputComplex->length/2;
        std::vector<double> returnVal2( middlePos , inputComplex->phas + inputComplex->length );
        returnVal2.insert( returnVal2.end(), inputComplex->phas, middlePos);
        for ( size_t i = 0; i < inputComplex->length; i++ )
            inputComplex->phas[i] = returnVal2[i];
    }

    virtual void doChunk( fvec_t *inputSimple, cvec_t *inputComplex ) override
    {
        swapInputWithIn(inputComplex);
        aubio_pvoc_rdo( pv, inputComplex, inputSimple);
        aubio_pitch_do (pitch, inputSimple, pitchOut);
        if ( pitchOut->data[0] < MIN_FREQ || pitchOut->data[0] > MAX_FREQ )
            return;

        getFormants( pitchOut->data[0], inputComplex);
        getSecondPitch( pitchOut->data[0], inputComplex, inputSimple );
        //void aubio_pvoc_rdo(aubio_pvoc_t *pv, cvec_t * fftgrain, fvec_t *out);

        colSize++;
    }

private:

    int selectedFormant;
    aubio_pitch_t *pitch;
    aubio_pvoc_t*  pv;
    fvec_t *pitchOut;
};

#endif // F0FEATURES


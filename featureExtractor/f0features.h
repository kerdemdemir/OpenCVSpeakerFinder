#ifndef F0FEATURES
#define F0FEATURES

#include "featureExtractor/featureExtractor.h"
#include "general.h"
#include <aubio.h>

constexpr int FORMANT_COUNT = 5;
constexpr int STATE_COUNT =  ((MAX_FREQ - MIN_FREQ) / JUMPSIZE);

class F0Features : public FeatureExtractor
{
public:

    F0Features( int formant )
    {
        selectedFormant = formant;
        char cStr[] = "default";
        samples = cv::Mat(  1, 1 , CV_64FC1 );
        pitchOut = new_fvec (1); // output candidate
        pitch = new_aubio_pitch (cStr, win_s, hopSize, sampleRate);
    }

    ~F0Features()
    {
        del_aubio_pitch (pitch);
        del_fvec (pitchOut);
    }

    int findTheBestValley( cvec_t* inputComplex, int index )
    {
        auto tempIndexBack = index;
        if ( inputComplex->norm[tempIndexBack-1] > inputComplex->norm[tempIndexBack])
            tempIndexBack--;

        auto tempIndexForward = index;
        if ( inputComplex->norm[tempIndexForward+1] > inputComplex->norm[tempIndexForward])
            tempIndexForward++;

        return inputComplex->norm[tempIndexBack] > inputComplex->norm[tempIndexForward] ? tempIndexBack : tempIndexForward;
    }

    void getFormants( double f0, cvec_t* inputComplex )
    {
       double freqStep = sampleRate / win_s;
       std::array< std::pair<double, double> , FORMANT_COUNT> formants;
       formants[0].first = f0;
       for ( int curFreq = f0; curFreq < FORMANT_COUNT * 1000; curFreq += f0 )
       {
            if ( curFreq < 1000 )
                continue;

            int formant = curFreq / 1000;
            int formantIndex = curFreq / freqStep;
            //formantIndex = findTheBestValley(inputComplex, formantIndex);
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


    virtual void doChunk( fvec_t *inputSimple, cvec_t *inputComplex ) override
    {
        aubio_pitch_do (pitch, inputSimple, pitchOut);
        if ( pitchOut->data[0] < MIN_FREQ || pitchOut->data[0] > MAX_FREQ )
            return;

        getFormants( pitchOut->data[0], inputComplex);

        //void aubio_pvoc_rdo(aubio_pvoc_t *pv, cvec_t * fftgrain, fvec_t *out);

        colSize++;
    }

private:

    int selectedFormant;
    aubio_pitch_t *pitch;
    fvec_t *pitchOut;
};

#endif // F0FEATURES


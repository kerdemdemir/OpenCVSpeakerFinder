#ifndef PYINF0FEATURE
#define PYINF0FEATURE

#include "featureExtractor/featureExtractor.h"
#include "general.h"
#include <libpyinc.h>


class PYINF0 : public FeatureExtractor
{
public:

    PYINF0( int formant )
    {
        lastPerson = 0;
        selectedFormant = formant;
        char cStr[] = "default";

        if ( selectedFormant <= 0 )
            samples = cv::Mat(  1, 1 , CV_64FC1 );
        else
            samples = cv::Mat(  1, FORMANT_COUNT , CV_64FC1 );

        pyinc_init( 16000, win_s, hopSize );
        pyinc_set_cut_off( 0.7 );
    }

    ~PYINF0()
    {
        pyinc_clear();
    }


    virtual DataType2D& getFeatures() override
    {
        return samples;
    }

    virtual void filefinished( const std::string& fileName )
    {
        int state = fileName2State(fileName);
        if ( state != lastPerson)
        {
            lastPerson = state;
            pyinc_clear();
        }
        pyinc_set_cut_off( 0.7 );
        FeatureExtractor::filefinished(fileName);
    }

    void setFormant( int finalPitch, cvec_t *inputComplex )
    {
        double freqStep = sampleRate / win_s;
        std::array< std::pair<double, double> , FORMANT_COUNT> formants;
        formants[0].first = finalPitch;
        for ( int curFreq = finalPitch; curFreq < FORMANT_COUNT * 1000; curFreq += finalPitch )
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

        samples.at<double>(colSize, 0) = (formants[selectedFormant].first) / ( 25 );
    }

    double getF0Amplitude( double f0, cvec_t* inputComplex )
    {
       double freqStep = sampleRate / win_s;
       int formantIndex = f0 / freqStep;
       return inputComplex->norm[formantIndex];
    }

    virtual void doChunk( fvec_t *inputSimple, cvec_t *inputComplex ) override
    {
        pyinc_feed( inputSimple->data, inputSimple->length );
        auto pitches = pyinc_get_pitches();
        double f0AmpResult = INT32_MIN;
        int finalPitch = 0;
        while ( pitches.begin != pitches.end )
        {
            auto curPitch = *pitches.begin++;
            if ( curPitch < MIN_FREQ || curPitch > MAX_FREQ )
                continue;

            auto curF0Amp = getF0Amplitude( curPitch, inputComplex);
            if ( curF0Amp > f0AmpResult )
            {
                finalPitch = curPitch;
                f0AmpResult = curF0Amp;
            }
        }

        if ( finalPitch > MIN_FREQ && finalPitch < MAX_FREQ )
        {
            if ( selectedFormant > 0 )
                setFormant(finalPitch, inputComplex);
            else
                samples.at<double>(colSize, 0) = (finalPitch - MIN_FREQ) / JUMPSIZE;

            colSize++;
        }


    }

private:

    int lastPerson;
    int selectedFormant;
    aubio_pitch_t *pitch;
};



#endif // PYINF0FEATURE


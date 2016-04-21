#ifndef TRANIERLIST
#define TRANIERLIST

#include "modelbase.h"
#include "featureExtractor/featurelist.h"
#include "featureExtractor/f0features.h"
#include "featureExtractor/MFCCFeatures.h"
#include "featureExtractor/f0highlevelfeatures.h"
#include "mlModel/gmmModel.h"
#include "mlModel/pitchgrams.h"
#include "mlModel/tranierlist.h"
#include <QElapsedTimer>

class TrainerComposer : public ModelBase
{
public:

    TrainerComposer()
    {
        modelName = "Fusion";
        init();
    }

    virtual void predict( const std::string& fileName )
    {
        featureList.start(fileName);

        QElapsedTimer timer;
        for ( auto model : modelList)
        {
            try
            {
                timer.restart();
                model->predict( fileName );
                model->predictionTime += timer.elapsed();
            }
            catch ( ... )
            {
                std::cout << "Exp caught while predicting " << std::endl;
                continue;
            }
        }
    }

    virtual void feed( const std::string& fileName )
    {        
        if (featureList.start(fileName) == -1)
            return;

        for ( auto model : modelList)
        {
            if ( !model->isLoad )
                model->feed( fileName );
        }
    }

    virtual void train ()
    {
        QElapsedTimer timer;
        for ( auto model : modelList)
        {
            try
            {
                timer.restart();
                if ( !model->isLoad )
                {
                    model->train();
                    model->save();
                }
                model->miliSecTrainTime += timer.elapsed();
            }
            catch ( ... )
            {
                std::cout << "Exp caught while training " << std::endl;
                continue;
            }
        }
        featureList.trainingOver();
    }

    virtual void save()
    {
        for ( auto model : modelList)
            model->save();
    }

    virtual void load()
    {
        for ( auto model : modelList)
        {
            if ( model->isLoad )
                model->load();
        }
    }

    bool isAllLoaded()
    {
        bool isAllLoaded = true;
        for ( auto model : modelList)
        {
            if ( !model->isLoad )
                isAllLoaded = false;
        }
        return isAllLoaded;
    }

    void addModel( std::shared_ptr<ModelBase> model )
    {
        modelList.push_back( model );
    }

    void init()
    {
        auto mfccFeaturePtr = std::make_shared<MFCCFeatures>();
        auto gmmModel = std::make_shared<GMMModel>("MFCC");
        featureList.addExtractor(mfccFeaturePtr);
        gmmModel->setFeature( mfccFeaturePtr );
        gmmModel->isLoad = false;
        addModel(gmmModel);

        auto F0FeaturePtr = std::make_shared<F0Features>(-1);
        auto gmmF0Model = std::make_shared<GMMModel>("Formant");
        featureList.addExtractor(F0FeaturePtr);
        gmmF0Model->setFeature( F0FeaturePtr );
        gmmF0Model->isLoad = false;
        addModel(gmmF0Model);

//        auto highLevelFeaturePtr = std::make_shared<F0HighLevelFeatures>(2,2);
//        auto gmmF0Model = std::make_shared<GMMModel>("HighLevelFormant");
//        featureList.addExtractor(highLevelFeaturePtr);
//        gmmF0Model->setFeature( highLevelFeaturePtr );
//        gmmF0Model->isLoad = false;
//        addModel(gmmF0Model);


        auto pitchGramRunnerModel = std::make_shared<PitchGramModel>(3, "F0");
        pitchGramRunnerModel->isLoad = false;
        pitchGramRunnerModel->setFeature( F0FeaturePtr );
        addModel(pitchGramRunnerModel);
    }

    std::vector<std::shared_ptr<ModelBase>>&
    getModels()
    {
        return modelList;
    }

private:

    FeatureList featureList;
    std::vector<std::shared_ptr<ModelBase>> modelList;
};


void train( TrainerComposer& trainer, const std::string& testFilePath )
{

    trainer.load();
    if ( trainer.isAllLoaded() )
        return;

    for (auto& elem : getFileNames(testFilePath))
    {
        std::cout << "Traning: " << elem.first << std::endl;
        trainer.feed(elem.first);
    }
    trainer.train();


    std::cout << "Training is over " << std::endl;
}

void process( TrainerComposer& trainer, const  std::string& testFilePath )
{
    for (auto& elem : getFileNames(testFilePath))
    {
        std::cout << elem.first << std::endl;
        trainer.predict( elem.first );
    }
}



#endif // TRANIERLIST


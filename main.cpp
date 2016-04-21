#include <vector>
#include <iostream>
#include <string>
#include <QDir>
#include <memory>

//#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "general.h"
#include "mlModel/tranierlist.h"
#include "statmanager.h"

using namespace std;



int main ()
{
    std::string trainPath("D:/speakerWavs/train1");
    //std::string trainPath("D:/speakerWavs/newGuysTrain");
    //

    //std::string testPath("D:/speakerWavs/temp");

    std::string testPath("D:/speakerWavs/testAndValidation");

    //getFileNames(trainPath);

    TrainerComposer trainer;
    train(trainer, trainPath);
    process(trainer, testPath);

    StatEvaluator stats;
    stats.parseResults( trainer.getModels() );

  return 0;
}

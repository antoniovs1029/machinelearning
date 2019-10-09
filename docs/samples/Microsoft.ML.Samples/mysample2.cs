using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ConsoleApp10ML.Model
{
    public class ModelInput
    {
        [ColumnName("Label"), LoadColumn(0)]
        public string Label { get; set; }


        [ColumnName("Title"), LoadColumn(1)]
        public string Title { get; set; }


        [ColumnName("Url"), LoadColumn(2)]
        public string Url { get; set; }


        [ColumnName("ImagePath"), LoadColumn(3)]
        public string ImagePath { get; set; }


    }

    public class ModelOutput
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public String Prediction { get; set; }
        public float[] Score { get; set; }
    }

    public class ConsumeModel
    {
        // For more info on consuming ML.NET models, visit https://aka.ms/model-builder-consume
        // Method for consuming model in your app
        public static ModelOutput Predict(ModelInput input)
        {

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            // var original = "C:\\Users\\anvelazq\\Desktop\\issue04\\originalmodel.zip";
            var modified = "C:\\Users\\anvelazq\\Desktop\\issue04\\modifiedmodel2.zip";

            ITransformer mlModel = mlContext.Model.Load(modified, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use model to make prediction on input data
            ModelOutput result = predEngine.Predict(input);
            return result;
        }
    }
}

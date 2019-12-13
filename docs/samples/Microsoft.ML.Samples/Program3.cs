using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MyData
{
    class MyData
    {
        public static void Leak(string[] args)
        {
            //var mlContext = new MLContext();


            // var preview = data.Preview(50);

            var stopwatch = Stopwatch.StartNew();
            var process = Process.GetCurrentProcess();

            var model = CreateModel();

            // var model = mlContext.Model.Load(@"C:\Users\anvelazq\Desktop\issue15\model.zip", out var inputschema);

            Console.WriteLine($"Memory in use after returning transformer: {Process.GetCurrentProcess().WorkingSet64 / 1000000000.0} GB");

            GC.Collect();

            //mlContext.Model.Save(model, data.Schema, @"C:\Users\anvelazq\Desktop\issue15\model.zip");

            // Console.WriteLine($"Elapsed: {stopwatch.ElapsedMilliseconds / 1000.0} seconds");
            Console.WriteLine($"Memory in use after calling GC: {Process.GetCurrentProcess().WorkingSet64 / 1000000000.0} GB");
            Console.WriteLine("Done");
        }

        static ITransformer CreateModel()
        {
            var mlContext = new MLContext();
            #region textloader
            var textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = new TextLoader.Column[]
                {
                    new TextLoader.Column("m:Index", DataKind.String, 0),
                    new TextLoader.Column("m:Column1", DataKind.String, 1),
                    new TextLoader.Column("m:Column2", DataKind.String, 2),
                },
                HasHeader = true,
                Separators = new char[] { ',' }
            });

            #endregion
            var data = textLoader.Load(@"C:\Users\anvelazq\Desktop\issue15\mydata.csv");

            var textColumns = new string[]
            {
                // "m:Index",
                "m:Column1",
                // "m:Column2",
            };

            var featurizers = new List<TextFeaturizingEstimator>();
            foreach (var textColumn in textColumns)
            {
                var featurizer = mlContext.Transforms.Text.FeaturizeText(textColumn, new TextFeaturizingEstimator.Options()
                {
                    CharFeatureExtractor = null,
                    WordFeatureExtractor = new WordBagEstimator.Options()
                    {
                        NgramLength = 2,
                        MaximumNgramsCount = new int[] { 200000 }
                    }
                });
                featurizers.Add(featurizer);
            }

            IEstimator<ITransformer> pipeline = featurizers.First();
            foreach (var featurizer in featurizers.Skip(1))
            {
                pipeline = pipeline.Append(featurizer);
            }

            var transformer = pipeline.Fit(data);

            var process = Process.GetCurrentProcess();
            Console.WriteLine($"Memory in use: {Process.GetCurrentProcess().WorkingSet64 / 1000000000.0} GB");
            Console.WriteLine("Returning Transformer");

            var data2 = transformer.Transform(mlContext.Data.TakeRows(data, 10000));
            var preview2 = data2.Preview(50);

            return transformer;
        }
    }
}


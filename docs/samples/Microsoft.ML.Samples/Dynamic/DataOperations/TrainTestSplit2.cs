using System;
using System.Collections.Generic;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    /// <summary>
    /// Sample class showing how to use TrainTestSplit.
    /// </summary>
    public static class TrainTestSplit2
    {
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline.
            var mlContext = new MLContext();

            // Generate some data points.
            var examples = GenerateRandomDataPoints(1500);

            // Convert the examples list to an IDataView object, which is consumable
            // by ML.NET API.
            var dataview = mlContext.Data.LoadFromEnumerable(examples);

            // Leave out 10% of the dataset for testing.For some types of problems,
            // for example for ranking or anomaly detection, we must ensure that the
            // split leaves the rows with the same value in a particular column, in
            // one of the splits. So below, we specify Group column as the column
            // containing the sampling keys. Notice how keeping the rows with the
            // same value in the Group column overrides the testFraction definition. 
            //var split = mlContext.Data
            //    .TrainTestSplit(dataview, testFraction: 0.1,
            //    samplingKeyColumnName: "Group");

            //var trainSet = mlContext.Data
            //    .CreateEnumerable<DataPoint>(split.TrainSet, reuseRowObject: false);

            //var testSet = mlContext.Data
            //    .CreateEnumerable<DataPoint>(split.TestSet,reuseRowObject: false);

            //PrintPreviewRows(trainSet, testSet);

            //  The data in the Train split.
            //  [Group, 1], [Features, 0.8173254]
            //  [Group, 1], [Features, 0.5581612]
            //  [Group, 1], [Features, 0.5588848]
            //  [Group, 1], [Features, 0.4421779]
            //  [Group, 1], [Features, 0.2737045]

            //  The data in the Test split.
            //  [Group, 0], [Features, 0.7262433]
            //  [Group, 0], [Features, 0.7680227]
            //  [Group, 0], [Features, 0.2060332]
            //  [Group, 0], [Features, 0.9060271]
            //  [Group, 0], [Features, 0.9775497]

            // Example of a split without specifying a sampling key column.
            dataview = mlContext.Transforms.Conversion.MapValueToKey("MyLabelKey", "MyLabel").Fit(dataview).Transform(dataview);
            Console.WriteLine("########### Original Distribution");
            PrintClassDistributions(dataview, "MyLabelKey");

            // ########### Using current TrainTestSplit
            Console.WriteLine("\n########### Using current TrainTestSplit");
            var split = mlContext.Data.TrainTestSplit(dataview, 0.25);

            var trainSet = mlContext.Data
                .CreateEnumerable<OutDataPoint>(split.TrainSet, reuseRowObject: false);

            var testSet = mlContext.Data
                .CreateEnumerable<OutDataPoint>(split.TestSet, reuseRowObject: false);
            Console.WriteLine("TrainSet");
            PrintClassDistributions(split.TrainSet, "MyLabelKey");
            Console.WriteLine("TestSet");
            PrintClassDistributions(split.TestSet, "MyLabelKey");

            // MYTODO: After here, the 'balance' and 'stratificationcolumn' refer to create a balanced or stratified TRAIN set, whereas the TESTfraction is provided
            // Should this API be changed?

            // ########### Stratified TrainTestSplit
            Console.WriteLine("\n########### Using Stratified TrainTestSplit");
            var split2 = mlContext.Data.StratifiedTrainTestSplit(dataview, "MyLabelKey", 0.25);

            var trainSet2 = mlContext.Data
                .CreateEnumerable<OutDataPoint>(split2.TrainSet, reuseRowObject: false);

            var testSet2 = mlContext.Data
                .CreateEnumerable<OutDataPoint>(split2.TestSet, reuseRowObject: false);
            Console.WriteLine("TrainSet");
            PrintClassDistributions(split2.TrainSet, "MyLabelKey");
            Console.WriteLine("TestSet");
            PrintClassDistributions(split2.TestSet, "MyLabelKey");

            // ########### Autobalanced Distribution
            Console.WriteLine("\n########### Using Balanced TrainTestSplit"); 
            var split3 = mlContext.Data.StratifiedTrainTestSplit(dataview, "MyLabelKey", balance: true, testFraction: 0.75); // MYTODO: testFraction is high, because it's difficult to form a big balanced train set (not enough samples)

            var trainSet3 = mlContext.Data
                .CreateEnumerable<OutDataPoint>(split3.TrainSet, reuseRowObject: false);

            var testSet3 = mlContext.Data
                .CreateEnumerable<OutDataPoint>(split3.TestSet, reuseRowObject: false);
            Console.WriteLine("TrainSet");
            PrintClassDistributions(split3.TrainSet, "MyLabelKey");
            Console.WriteLine("TestSet");
            PrintClassDistributions(split3.TestSet, "MyLabelKey");

            Console.WriteLine("END");

            //PrintPreviewRows(trainSet, testSet);

            // The data in the Train split.
            // [Group, 0], [Features, 0.7262433]
            // [Group, 1], [Features, 0.8173254]
            // [Group, 0], [Features, 0.7680227]
            // [Group, 1], [Features, 0.5581612]
            // [Group, 0], [Features, 0.2060332]
            // [Group, 1], [Features, 0.4421779]
            // [Group, 0], [Features, 0.9775497]
            // [Group, 1], [Features, 0.2737045]

            // The data in the Test split.
            // [Group, 1], [Features, 0.5588848]
            // [Group, 0], [Features, 0.9060271]

        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)

        {
            var random = new Random(seed);

            var strings = new[] { "cat", "dog", "cow", "horse" };

            for (int i = 0; i < count; i++)
            {
                yield return new DataPoint
                {
                    Group = i % 2,
                    MyLabel = strings[RandomIndex(random.NextDouble())],
                    // Create random features that are correlated with label.
                    Features = (float)random.NextDouble()
                };
            }
                
            //for(int i = 0; i < count/100; i++)
            //{
            //    yield return new DataPoint
            //    {
            //        Group = i /2,
            //        MyLabel = "sheep",
            //        Features = (float)random.NextDouble()
            //    };
            //}
        }

        private static int RandomIndex(double randomDouble)
        {
            if(randomDouble < .10)
            {
                return 0;
            }
            if(randomDouble < .30)
            {
                return 1;
            }
            if(randomDouble < .60)
            {
                return 2;
            }

            return 3;
        }

        // Example with label and group column. A data set is a collection of such
        // examples.
        private class DataPoint
        {
            public float Group { get; set; }

            public string MyLabel { get; set; }

            public float Features { get; set; }
        }

        private class OutDataPoint : DataPoint
        {
            public uint MyLabelKey { get; set; }
        }

        // print helper
        private static void PrintPreviewRows(IEnumerable<OutDataPoint> trainSet,
            IEnumerable<OutDataPoint> testSet)

        {

            Console.WriteLine($"The data in the Train split.");
            foreach (var row in trainSet)
                Console.WriteLine($"{row.Group}, {row.Features}, {row.MyLabel}, {row.MyLabelKey}");

            Console.WriteLine($"\nThe data in the Test split.");
            foreach (var row in testSet)
                Console.WriteLine($"{row.Group}, {row.Features}, {row.MyLabel}, {row.MyLabelKey}");
        }

        private static void PrintClassDistributions(IDataView data, string classesColumnName)
        {
            var col = data.Schema[classesColumnName];
            var counters = new Dictionary<uint, int>();
            int nrows = 0;
            using (var cursor = data.GetRowCursor(new[] { col }))
            {
                uint intValue = default;
                var intGetter = cursor.GetGetter<uint>(col);
                while (cursor.MoveNext())
                {
                    intGetter(ref intValue);
                    if (!counters.ContainsKey(intValue))
                    {
                        counters.Add(intValue, 0);
                    }

                    counters[intValue] += 1;
                    nrows++;
                }
            }

            Console.WriteLine($"Number of rows: {nrows}");
            int count = 0;
            for(int i = 1; i <= counters.Count; i++)
            {
                counters.TryGetValue((uint)i, out count);
                Console.WriteLine($"Class #{i} - Count: {count} - {(double)count* 100 / nrows: 0.00}%");
            }
        }
    }
}

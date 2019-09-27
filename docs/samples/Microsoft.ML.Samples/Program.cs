using System;
using System.Reflection;
using Samples.Dynamic;
using Samples.Dynamic.Trainers.BinaryClassification;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll();

        internal static void RunAll()
        {
            PermutationFeatureImportance2.Example();
        }
    }
}

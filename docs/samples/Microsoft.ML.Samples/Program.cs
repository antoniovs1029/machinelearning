using System;
using System.Reflection;
using Samples.Dynamic;
using ConsoleApp10ML.ConsoleApp;

namespace Microsoft.ML.Samples
{
    public static class Program
    {
        public static void Main(string[] args) => RunAll();

        internal static void RunAll()
        {
            ModelBuilder.CreateModel();
        }
    }
}

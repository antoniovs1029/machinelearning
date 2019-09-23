// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(typeof(RegressionPredictionTransformer<IPredictorProducing<float>>), typeof(RegressionPredictionTransformer), null, typeof(SignatureLoadModel),
    "", RegressionPredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{

    /// <summary>
    /// Base class for transformers with no feature column, or more than one feature columns.
    /// </summary>
    /// <typeparam name="TModel">The type of the model parameters used by this prediction transformer.</typeparam>
    public abstract class PredictionTransformerBase<TModel> : IPredictionTransformer<TModel>
        where TModel : class
    {
        /// <summary>
        /// The model.
        /// </summary>
        public TModel Model { get; }

        private protected IPredictor ModelAsPredictor => (IPredictor)Model;

        [BestFriend]
        private protected const string DirModel = "Model";
        [BestFriend]
        private protected const string DirTransSchema = "TrainSchema";
        [BestFriend]
        private protected readonly IHost Host;
        [BestFriend]
        private protected ISchemaBindableMapper BindableMapper;
        [BestFriend]
        internal DataViewSchema TrainSchema;

        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper(DataViewSchema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => true;

        /// <summary>
        /// This class is more or less a thin wrapper over the <see cref="IDataScorerTransform"/> implementing
        /// <see cref="RowToRowScorerBase"/>, which publicly is a deprecated concept as far as the public API is
        /// concerned. Nonetheless, until we move all internal infrastructure to be truely transform based, we
        /// retain this as a wrapper. Even though it is mutable, subclasses of this should set this only in
        /// their constructor.
        /// </summary>
        [BestFriend]
        private protected RowToRowScorerBase Scorer { get; set; }

        [BestFriend]
        private protected PredictionTransformerBase(IHost host, TModel model, DataViewSchema trainSchema)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;

            Host.CheckValue(model, nameof(model));
            Host.CheckParam(model is IPredictor, nameof(model));
            Model = model;

            Host.CheckValue(trainSchema, nameof(trainSchema));
            TrainSchema = trainSchema;
        }

        [BestFriend]
        private protected PredictionTransformerBase(IHost host, ModelLoadContext ctx, TModel model)
        {
            Host = host;

            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // id of string: feature column.

            // ctx.LoadModel<TModel, SignatureLoadModel>(host, out TModel model, DirModel);

            Model = model;

            // Clone the stream with the schema into memory.
            var ms = new MemoryStream();
            ctx.TryLoadBinaryStream(DirTransSchema, reader =>
            {
                reader.BaseStream.CopyTo(ms);
            });

            ms.Position = 0;
            var loader = new BinaryLoader(host, new BinaryLoader.Arguments(), ms);
            TrainSchema = loader.Schema;
        }

        /// <summary>
        /// Gets the output schema resulting from the <see cref="Transform(IDataView)"/>
        /// </summary>
        /// <param name="inputSchema">The <see cref="DataViewSchema"/> of the input data.</param>
        /// <returns>The resulting <see cref="DataViewSchema"/>.</returns>
        public abstract DataViewSchema GetOutputSchema(DataViewSchema inputSchema);

        /// <summary>
        /// Transforms the input data.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <returns>The transformed <see cref="IDataView"/></returns>
        public IDataView Transform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return Scorer.ApplyToData(Host, input);
        }

        /// <summary>
        /// Gets a IRowToRowMapper instance.
        /// </summary>
        /// <param name="inputSchema"></param>
        /// <returns></returns>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return (IRowToRowMapper)Scorer.ApplyToData(Host, new EmptyDataView(Host, inputSchema));
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected abstract void SaveModel(ModelSaveContext ctx);

        [BestFriend]
        private protected void SaveModelCore(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // <base info>
            // stream: empty data view that contains train schema.

            ctx.SaveModel(Model, DirModel);
            ctx.SaveBinaryStream(DirTransSchema, writer =>
            {
                using (var ch = Host.Start("Saving train schema"))
                {
                    var saver = new BinarySaver(Host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(Host, TrainSchema), writer.BaseStream);
                }
            });
        }
    }

    /// <summary>
    /// The base class for all the transformers implementing the <see cref="ISingleFeaturePredictionTransformer{TModel}"/>.
    /// Those are all the transformers that work with one feature column.
    /// </summary>
    /// <typeparam name="TModel">The model used to transform the data.</typeparam>
    public abstract class SingleFeaturePredictionTransformerBase<TModel> : PredictionTransformerBase<TModel>, ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        /// <summary>
        /// The name of the feature column used by the prediction transformer.
        /// </summary>
        public string FeatureColumnName { get; }

        /// <summary>
        /// The type of the prediction transformer
        /// </summary>
        public DataViewType FeatureColumnType { get; }

        /// <summary>
        /// Initializes a new reference of <see cref="SingleFeaturePredictionTransformerBase{TModel}"/>.
        /// </summary>
        /// <param name="host">The local instance of <see cref="IHost"/>.</param>
        /// <param name="model">The model used for scoring.</param>
        /// <param name="trainSchema">The schema of the training data.</param>
        /// <param name="featureColumn">The feature column name.</param>
        private protected SingleFeaturePredictionTransformerBase(IHost host, TModel model, DataViewSchema trainSchema, string featureColumn)
            : base(host, model, trainSchema)
        {
            FeatureColumnName = featureColumn;
            if (featureColumn == null)
                FeatureColumnType = null;
            else if (!trainSchema.TryGetColumnIndex(featureColumn, out int col))
                throw Host.ExceptSchemaMismatch(nameof(featureColumn), "feature", featureColumn);
            else
                FeatureColumnType = trainSchema[col].Type;

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, ModelAsPredictor);
        }

        private protected SingleFeaturePredictionTransformerBase(IHost host, ModelLoadContext ctx, TModel model)
            : base(host, ctx, model)
        {
            FeatureColumnName = ctx.LoadStringOrNull();

            if (FeatureColumnName == null)
                FeatureColumnType = null;
            else if (!TrainSchema.TryGetColumnIndex(FeatureColumnName, out int col))
                throw Host.ExceptSchemaMismatch(nameof(FeatureColumnName), "feature", FeatureColumnName);
            else
                FeatureColumnType = TrainSchema[col].Type;

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, ModelAsPredictor);
        }

        /// <summary>
        ///  Schema propagation for this prediction transformer.
        /// </summary>
        /// <param name="inputSchema">The input schema to attempt to map.</param>
        /// <returns>The output schema of the data, given an input schema like <paramref name="inputSchema"/>.</returns>
        public sealed override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (FeatureColumnName != null)
            {
                if (!inputSchema.TryGetColumnIndex(FeatureColumnName, out int col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", FeatureColumnName);
                if (!inputSchema[col].Type.Equals(FeatureColumnType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", FeatureColumnName, FeatureColumnType.ToString(), inputSchema[col].Type.ToString());
            }

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        private protected sealed override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        private protected virtual void SaveCore(ModelSaveContext ctx)
        {
            SaveModelCore(ctx);
            ctx.SaveStringOrNull(FeatureColumnName);
        }

        private protected GenericScorer GetGenericScorer()
        {
            var schema = new RoleMappedSchema(TrainSchema, null, FeatureColumnName);
            return new GenericScorer(Host, new GenericScorer.Arguments(), new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }
    }

    /// <summary>
    /// Base class for the <see cref="ISingleFeaturePredictionTransformer{TModel}"/> working on regression tasks.
    /// </summary>
    /// <typeparam name="TModel">An implementation of the <see cref="IPredictorProducing{TResult}"/></typeparam>
    public sealed class RegressionPredictionTransformer<TModel> : SingleFeaturePredictionTransformerBase<TModel>
        where TModel : class
    {
        [BestFriend]
        internal RegressionPredictionTransformer(IHostEnvironment env, TModel model, DataViewSchema inputSchema, string featureColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<TModel>)), model, inputSchema, featureColumn)
        {
            Scorer = GetGenericScorer();
        }

        internal RegressionPredictionTransformer(IHostEnvironment env, ModelLoadContext ctx, IHost host, TModel model)
            : base(host, ctx, model)
        {
            Scorer = GetGenericScorer();
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            base.SaveCore(ctx);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "REG PRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: RegressionPredictionTransformer.LoaderSignature,
                loaderAssemblyName: typeof(RegressionPredictionTransformer<>).Assembly.FullName);
        }
    }

    internal static class RegressionPredictionTransformer
    {
        public const string LoaderSignature = "RegressionPredXfer";

        public static RegressionPredictionTransformer<IPredictorProducing<float>> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            // new RegressionPredictionTransformer<IPredictorProducing<float>>(env, ctx);
            return Utils.MarshalInvoke(CreateType<IPredictorProducing<float>>, typeof(IPredictorProducing<float>), env, ctx);
        }

        public static RegressionPredictionTransformer<T> CreateType<T>(IHostEnvironment env, ModelLoadContext ctx) where T: class
        {
            var host = Contracts.CheckRef(env, nameof(env)).Register(nameof(RegressionPredictionTransformer<T>));
            ctx.LoadModel<T, SignatureLoadModel>(host, out T model, "Model"); // MYTODO: don't hardcode the DirModel
            return new RegressionPredictionTransformer<T>(env, ctx, host, model);
        }
    }
}

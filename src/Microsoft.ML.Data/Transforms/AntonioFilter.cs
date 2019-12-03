// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(AntonioFilter.Summary, typeof(AntonioFilter), typeof(AntonioFilter.Options), typeof(SignatureDataTransform),
    AntonioFilter.UserName, "AntonioFilter")]

[assembly: LoadableClass(AntonioFilter.Summary, typeof(AntonioFilter), null, typeof(SignatureLoadDataTransform),
    AntonioFilter.UserName, AntonioFilter.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    // REVIEW: Should we support filtering on multiple columns/vector typed columns?
    /// <summary>
    /// Filters a dataview on a column of type Single, Double or Key (contiguous).
    /// Keeps the values that are in the specified min/max range. NaNs are always filtered out.
    /// If the input is a Key type, the min/max are considered percentages of the number of values.
    /// </summary>
    [BestFriend]
    internal sealed class AntonioFilter : FilterBase
    {
        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Column", ShortName = "col", SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Column;

            [Argument(ArgumentType.Multiple, HelpText = "If true, keep the values that fall outside the range.")]
            public bool Complement;

            public Dictionary<uint, int> Dict; //MYTODO: Should I put ArgumenType in here and the other params?
        }

        // MYTODO: Update this descriptions and the GetVersionInfo
        public const string Summary = "Filters a dataview on a column of type Single, Double or Key (contiguous). Keeps the values that are in the specified min/max range. "
            + "NaNs are always filtered out. If the input is a Key type, the min/max are considered percentages of the number of values.";

        public const string LoaderSignature = "AntonioFilter";
        public const string UserName = "Range Filter";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RNGFILTR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(AntonioFilter).Assembly.FullName);
        }

        private const string RegistrationName = "AntonioFilter";

        private readonly int _index;
        private readonly DataViewType _type;
        private readonly bool _complement;
        private readonly Dictionary<uint, int> _dict;

        public AntonioFilter(IHostEnvironment env, Options options, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(options, nameof(options));

            var schema = Source.Schema;
            if (!schema.TryGetColumnIndex(options.Column, out _index))
                throw Host.ExceptUserArg(nameof(options.Column), "Source column '{0}' not found", options.Column);

            using (var ch = Host.Start("Checking parameters"))
            {
                _type = schema[_index].Type;
                if (!IsValidAntonioFilterColumnType(ch, _type))
                    throw ch.ExceptUserArg(nameof(options.Column), "Column '{0}' does not have compatible type", options.Column);

                _complement = options.Complement;
                _dict = new Dictionary<uint, int>(options.Dict); // MYTODO: Is there a better way to copy a dictionary to avoid modifying the original?
            }
        }

        private AntonioFilter(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // MYTODO: Implement this constructor for loading?
        }

        public static AntonioFilter Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new AntonioFilter(h, ctx, input));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // MYTODO: Implement Saving method?
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            // MYTODO: With the current implementation it shouldn't use parallel cursors because of the way it access the _dict
            // If parallel cursors were allowed, then the output dataviews are inconsistent, and typically contain more rows than expected
            // Particularlly, if parallel cursors were allowed, each cursor would have a diferent copy of _dict,
            // I don't know if there could be a solution that allows to have a shared dict among cursors AND that would allow the dict to restar
            // everytime the user wants to go again through the dataset.
            return false;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            Func<int, bool> inputPred = GetActive(predicate, out bool[] active);
            var inputCols = Source.Schema.Where(x => inputPred(x.Index));

            var input = Source.GetRowCursor(inputCols, rand);
            return CreateCursorCore(input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            // MYTODO: Is this still necessary if ShouldUseParallelCursors return false?

            Host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            Func<int, bool> inputPred = GetActive(predicate, out bool[] active);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var inputs = Source.GetRowCursorSet(inputCols, n, rand);
            Host.AssertNonEmpty(inputs);

            // No need to split if this is given 1 input cursor.
            var cursors = new DataViewRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = CreateCursorCore(inputs[i], active);
            return cursors;
        }

        private DataViewRowCursor CreateCursorCore(DataViewRowCursor input, bool[] active)
        {
            Host.Assert(_type is KeyDataViewType); // MYTODO: Currently, only support for KeyDataViewType is provided
            return RowCursorBase.CreateKeyRowCursor(this, input, active);
        }

        private Func<int, bool> GetActive(Func<int, bool> predicate, out bool[] active)
        {
            Host.AssertValue(predicate);
            active = new bool[Source.Schema.Count];
            bool[] activeInput = new bool[Source.Schema.Count];
            for (int i = 0; i < active.Length; i++)
                activeInput[i] = active[i] = predicate(i);
            activeInput[_index] = true;
            return col => activeInput[col];
        }

        public static bool IsValidAntonioFilterColumnType(IExceptionContext ectx, DataViewType type)
        {
            ectx.CheckValue(type, nameof(type));
            return type.GetKeyCount() > 0; // MYTODO: Currently, only support for KeyDataViewType is provided
        }

        private abstract class RowCursorBase : LinkedRowFilterCursorBase
        {
            protected readonly AntonioFilter Parent;

            protected RowCursorBase(AntonioFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent.Host, input, parent.OutputSchema, active)
            {
                Parent = parent;
            }

            protected abstract Delegate GetGetter();
            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.Check(0 <= column.Index && column.Index < Schema.Count);
                Ch.Check(IsColumnActive(column));

                if (column.Index != Parent._index)
                    return Input.GetGetter<TValue>(column);
                var fn = GetGetter() as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));

                return fn;
            }

            public static DataViewRowCursor CreateKeyRowCursor(AntonioFilter filter, DataViewRowCursor input, bool[] active)
            {
                Contracts.Assert(filter._type is KeyDataViewType);
                Func<AntonioFilter, DataViewRowCursor, bool[], DataViewRowCursor> del = CreateKeyRowCursor<int>;
                var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(filter._type.RawType);
                return (DataViewRowCursor)methodInfo.Invoke(null, new object[] { filter, input, active });
            }

            private static DataViewRowCursor CreateKeyRowCursor<TSrc>(AntonioFilter filter, DataViewRowCursor input, bool[] active)
            {
                Contracts.Assert(filter._type is KeyDataViewType);
                return new KeyRowCursor<TSrc>(filter, input, active);
            }
        }

        private sealed class KeyRowCursor<T> : RowCursorBase
        {
            private readonly ValueGetter<T> _srcGetter;
            private readonly ValueGetter<T> _getter;
            private T _value;
            private readonly ValueMapper<T, ulong> _conv;
            private readonly ulong _count;
            private readonly Dictionary<uint, int> _dict;

            public KeyRowCursor(AntonioFilter parent, DataViewRowCursor input, bool[] active)
                : base(parent, input, active)
            {
                Ch.Assert(Parent._type.GetKeyCount() > 0);
                _count = Parent._type.GetKeyCount();
                _srcGetter = Input.GetGetter<T>(Input.Schema[Parent._index]);
                _getter =
                    (ref T dst) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        dst = _value;
                    };
                bool identity;
                _conv = Data.Conversion.Conversions.Instance.GetStandardConversion<T, ulong>(Parent._type, NumberDataViewType.UInt64, out identity);
                _dict = new Dictionary<uint, int>(Parent._dict); // MYTODO: This way the dict is preserved even if the data is accessed multiple times... is it okay to copy it always, tho?
            }

            protected override Delegate GetGetter()
            {
                Ch.Assert(Parent._type is KeyDataViewType);
                return _getter;
            }

            protected override bool Accept()
            {
                Ch.Assert(Parent._type is KeyDataViewType);
                _srcGetter(ref _value);
                ulong value = 0;
                _conv(in _value, ref value);
                if (value == 0 || value > _count)
                    return false;

                if (_dict[(uint) value] > 0)
                {
                    _dict[(uint)value] -= 1;
                    return Parent._complement ? false : true;
                }
                return Parent._complement ? true : false;
            }
        }
    }
}

import unittest

import numpy as np
import pandas as pd

from forestgap.utils import (
    apply_missingness,
    missing_prep_transform,
    prepare_missing_data,
    remove_mar,
    remove_mcar,
    remove_mnar,
    train_test,
)


def _make_df(n=100, random_state=0):
    """Small mixed DataFrame with a label column first."""
    rng = np.random.default_rng(random_state)
    return pd.DataFrame({
        "label": rng.integers(0, 3, size=n).astype(int),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
        "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
    })


def _make_numeric_df(n=100, random_state=0):
    rng = np.random.default_rng(random_state)
    return pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n) * 2 + 1,
        "c": rng.standard_normal(n),
    })


# ---------------------------------------------------------------------------
# missing_prep_transform
# ---------------------------------------------------------------------------

class TestMissingPrepTransform(unittest.TestCase):

    def test_standardize_train_only(self):
        df = _make_numeric_df()
        result = missing_prep_transform(df.copy())
        self.assertIsInstance(result, pd.DataFrame)
        np.testing.assert_allclose(result["a"].mean(), 0, atol=1e-10)
        np.testing.assert_allclose(result["a"].std(ddof=0), 1, atol=1e-10)

    def test_normalize_train_only(self):
        df = _make_numeric_df()
        result = missing_prep_transform(df.copy(), continuous_transform='normalize')
        self.assertGreaterEqual(result["a"].min(), 0.0)
        self.assertLessEqual(result["a"].max(), 1.0)

    def test_standardize_train_test(self):
        df = _make_numeric_df(n=100)
        train, test = df.iloc[:80].copy(), df.iloc[80:].copy()
        train_out, test_out = missing_prep_transform(train, test_data=test)
        self.assertEqual(train_out.shape[1], test_out.shape[1])
        # test set is scaled by the training scaler, so its mean need not be 0
        self.assertIsInstance(test_out, pd.DataFrame)

    def test_categorical_encode(self):
        df = _make_df()
        # Only keep categorical + a float column
        sub = df[["cat", "x1"]].copy()
        result = missing_prep_transform(sub, categorical_transform='encode')
        self.assertTrue(pd.api.types.is_integer_dtype(result["cat"]) or
                        pd.api.types.is_numeric_dtype(result["cat"]))

    def test_categorical_encode_train_test_categories_aligned(self):
        """Test data categories are mapped using training categories, not inferred."""
        train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c", "a"])})
        test = pd.DataFrame({"cat": pd.Categorical(["b", "c"])})
        train_out, test_out = missing_prep_transform(
            train.copy(), test_data=test.copy(), categorical_transform='encode'
        )
        # Both should be numeric with no AttributeError
        self.assertTrue(pd.api.types.is_numeric_dtype(train_out["cat"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(test_out["cat"]))

    def test_onehot_encoding(self):
        df = _make_df()[["cat", "x1"]].copy()
        result = missing_prep_transform(df, categorical_transform='onehot')
        self.assertGreater(result.shape[1], 2)

    def test_onehot_train_test_columns_aligned(self):
        train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"]), "x": [1.0, 2.0, 3.0]})
        test = pd.DataFrame({"cat": pd.Categorical(["a"]), "x": [1.5]})
        train_out, test_out = missing_prep_transform(
            train.copy(), test_data=test.copy(), categorical_transform='onehot'
        )
        self.assertEqual(list(train_out.columns), list(test_out.columns))

    def test_invalid_continuous_transform_raises(self):
        with self.assertRaises(ValueError):
            missing_prep_transform(_make_numeric_df(), continuous_transform='bad')

    def test_invalid_categorical_transform_raises(self):
        df = _make_df()[["cat", "x1"]].copy()
        with self.assertRaises(ValueError):
            missing_prep_transform(df, categorical_transform='bad')


# ---------------------------------------------------------------------------
# remove_mcar
# ---------------------------------------------------------------------------

class TestRemoveMCAR(unittest.TestCase):

    def test_introduces_missing_values(self):
        df = _make_numeric_df(n=500)
        result = remove_mcar(df, pct=0.20)
        total_missing = result.isna().sum().sum()
        self.assertGreater(total_missing, 0)

    def test_approximate_missing_rate(self):
        df = _make_numeric_df(n=1000)
        pct = 0.15
        result = remove_mcar(df, pct=pct, random_state=0)
        total_cells = df.shape[0] * df.shape[1]
        actual_pct = result.isna().sum().sum() / total_cells
        self.assertAlmostEqual(actual_pct, pct, delta=0.03)

    def test_does_not_mutate_input(self):
        df = _make_numeric_df()
        original = df.copy()
        remove_mcar(df, pct=0.10)
        pd.testing.assert_frame_equal(df, original)

    def test_reproducible_with_seed(self):
        df = _make_numeric_df(n=200)
        r1 = remove_mcar(df.copy(), pct=0.10, random_state=7)
        r2 = remove_mcar(df.copy(), pct=0.10, random_state=7)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_differ(self):
        df = _make_numeric_df(n=200)
        r1 = remove_mcar(df.copy(), pct=0.20, random_state=1)
        r2 = remove_mcar(df.copy(), pct=0.20, random_state=99)
        self.assertFalse(r1.isna().equals(r2.isna()))

    def test_integer_column_uses_pd_na(self):
        df = pd.DataFrame({"i": pd.array([1, 2, 3, 4, 5] * 40, dtype="Int64")})
        result = remove_mcar(df, pct=0.30, random_state=0)
        self.assertTrue(result["i"].isna().any())

    def test_categorical_column(self):
        df = pd.DataFrame({"c": pd.Categorical(["x", "y", "z"] * 50)})
        result = remove_mcar(df, pct=0.20, random_state=0)
        self.assertTrue(result["c"].isna().any())


# ---------------------------------------------------------------------------
# remove_mnar
# ---------------------------------------------------------------------------

class TestRemoveMNAR(unittest.TestCase):

    def test_introduces_missing_values(self):
        df = _make_numeric_df(n=500)
        result = remove_mnar(df, q=0.8, p=0.9, random_state=0)
        self.assertGreater(result.isna().sum().sum(), 0)

    def test_missing_concentrated_above_quantile(self):
        """Missing values should appear predominantly in the upper tail."""
        df = _make_numeric_df(n=2000)
        result = remove_mnar(df, q=0.8, p=1.0, random_state=0)
        threshold = df["a"].quantile(0.8)
        missing_mask = result["a"].isna()
        # All missings must have come from above the quantile
        self.assertTrue((df.loc[missing_mask, "a"] > threshold).all())

    def test_does_not_mutate_input(self):
        df = _make_numeric_df()
        original = df.copy()
        remove_mnar(df, q=0.7, p=0.5, random_state=0)
        pd.testing.assert_frame_equal(df, original)

    def test_invalid_q_raises(self):
        with self.assertRaises(ValueError):
            remove_mnar(_make_numeric_df(), q=1.5, p=0.5)

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            remove_mnar(_make_numeric_df(), q=0.8, p=1.5)

    def test_unknown_column_raises(self):
        with self.assertRaises(ValueError):
            remove_mnar(_make_numeric_df(), q=0.8, p=0.5, columns=["nonexistent"])

    def test_subset_columns(self):
        df = _make_numeric_df(n=500)
        result = remove_mnar(df, q=0.8, p=1.0, columns=["a"], random_state=0)
        self.assertGreater(result["a"].isna().sum(), 0)
        self.assertEqual(result["b"].isna().sum(), 0)
        self.assertEqual(result["c"].isna().sum(), 0)


# ---------------------------------------------------------------------------
# remove_mar
# ---------------------------------------------------------------------------

class TestRemoveMAR(unittest.TestCase):

    def test_introduces_missing_values(self):
        df = _make_numeric_df(n=500)
        # Force targets/conditionals so correlation_threshold doesn't matter
        result = remove_mar(df, q=0.5, p=0.8, targets=["a"], conditionals=["b"], random_state=0)
        self.assertGreater(result["a"].isna().sum(), 0)

    def test_does_not_mutate_input(self):
        df = _make_numeric_df()
        original = df.copy()
        remove_mar(df, q=0.5, p=0.5, targets=["a"], conditionals=["b"], random_state=0)
        pd.testing.assert_frame_equal(df, original)

    def test_conditional_column_unchanged(self):
        df = _make_numeric_df(n=500)
        result = remove_mar(df, q=0.5, p=0.9, targets=["a"], conditionals=["b"], random_state=0)
        self.assertEqual(result["b"].isna().sum(), 0)

    def test_invalid_q_raises(self):
        with self.assertRaises(ValueError):
            remove_mar(_make_numeric_df(), q=0, p=0.5)

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            remove_mar(_make_numeric_df(), q=0.5, p=-0.1)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            remove_mar(_make_numeric_df(), q=0.5, p=0.5,
                       targets=["a"], conditionals=["nonexistent"])

    def test_auto_detect_correlated_columns(self):
        """When targets/conditionals are None, correlated columns should be found."""
        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal(n)
        df = pd.DataFrame({"a": x, "b": x + rng.standard_normal(n) * 0.1, "c": rng.standard_normal(n)})
        result = remove_mar(df, q=0.5, p=0.8, random_state=0)
        # At least one of the correlated columns should have missings
        self.assertGreater(result[["a", "b"]].isna().sum().sum(), 0)

    def test_no_correlated_features_returns_unchanged(self):
        """When no correlated features found and targets/conditionals are None, return unchanged."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.standard_normal(200),
            "b": rng.standard_normal(200),
        })
        result = remove_mar(df, q=0.5, p=0.9, correlation_threshold=0.999, random_state=0)
        self.assertEqual(result.isna().sum().sum(), 0)


# ---------------------------------------------------------------------------
# train_test
# ---------------------------------------------------------------------------

class TestTrainTest(unittest.TestCase):

    def test_output_shapes(self):
        df = _make_df(n=100)
        x_train, x_test, y_train, y_test = train_test(df.copy())
        self.assertEqual(len(x_train) + len(x_test), 100)
        self.assertEqual(len(y_train) + len(y_test), 100)

    def test_label_column_removed_from_x(self):
        df = _make_df(n=100)
        x_train, x_test, y_train, y_test = train_test(df.copy())
        self.assertNotIn("label", x_train.columns)
        self.assertNotIn("label", x_test.columns)

    def test_does_not_mutate_input(self):
        df = _make_df(n=50)
        cols_before = list(df.columns)
        train_test(df)
        self.assertEqual(list(df.columns), cols_before)

    def test_reproducible_split(self):
        df = _make_df(n=100)
        r1 = train_test(df.copy(), random_state=0)
        r2 = train_test(df.copy(), random_state=0)
        pd.testing.assert_frame_equal(r1[0], r2[0])


class TestMissingnessWorkflow(unittest.TestCase):

    def test_apply_missingness_mcar(self):
        df = _make_numeric_df(n=300)
        out = apply_missingness(df, mechanism='mcar', percentage=0.2, random_state=0)
        self.assertGreater(out.isna().sum().sum(), 0)

    def test_apply_missingness_mnar(self):
        df = _make_numeric_df(n=300)
        out = apply_missingness(df, mechanism='mnar', percentage=0.7, q=0.8, random_state=0)
        self.assertGreater(out.isna().sum().sum(), 0)

    def test_apply_missingness_mar(self):
        df = _make_numeric_df(n=300)
        out = apply_missingness(
            df,
            mechanism='mar',
            percentage=0.8,
            q=0.5,
            targets=['a'],
            conditionals=['b'],
            random_state=0,
        )
        self.assertGreater(out['a'].isna().sum(), 0)

    def test_apply_missingness_invalid_mechanism_raises(self):
        with self.assertRaises(ValueError):
            apply_missingness(_make_numeric_df(), mechanism='bad', percentage=0.1)

    def test_prepare_missing_data_mcar(self):
        df = _make_df(n=200)
        x_train_missing, x_train, x_test, x_test_missing, y_train, y_test = prepare_missing_data(
            df,
            mechanism='mcar',
            percentage=0.2,
            random_state=0,
        )
        self.assertGreater(x_train_missing.isna().sum().sum(), 0)
        self.assertGreater(x_test_missing.isna().sum().sum(), 0)
        self.assertEqual(x_train.shape[1], x_train_missing.shape[1])

    def test_prepare_missing_data_mnar(self):
        df = _make_df(n=200)
        x_train_missing, x_train, x_test, x_test_missing, y_train, y_test = prepare_missing_data(
            df,
            mechanism='mnar',
            percentage=0.7,
            q=0.8,
            random_state=0,
        )
        self.assertGreater(x_train_missing.isna().sum().sum(), 0)

    def test_prepare_missing_data_returns_six_values(self):
        df = _make_df(n=100)
        result = prepare_missing_data(df.copy(), mechanism='mcar', percentage=0.10)
        self.assertEqual(len(result), 6)

    def test_prepare_missing_data_missing_values_in_train_and_test(self):
        df = _make_df(n=200)
        x_train_missing, x_train, x_test, x_test_missing, y_train, y_test = prepare_missing_data(
            df.copy(), mechanism='mcar', percentage=0.20
        )
        self.assertGreater(x_train_missing.isna().sum().sum(), 0)
        self.assertGreater(x_test_missing.isna().sum().sum(), 0)
        self.assertEqual(x_train.isna().sum().sum(), 0)

    def test_prepare_missing_data_empty_df_returns_six_nones(self):
        df = pd.DataFrame()
        result = prepare_missing_data(df, mechanism='mcar', percentage=0.10)
        self.assertEqual(len(result), 6)
        for val in result[1:]:
            self.assertIsNone(val)


# ---------------------------------------------------------------------------
# numpy array support
# ---------------------------------------------------------------------------

def _make_array(n=100, ncols=3, random_state=0):
    rng = np.random.default_rng(random_state)
    return rng.standard_normal((n, ncols))


class TestMissingPrepTransformArray(unittest.TestCase):

    def test_returns_ndarray(self):
        arr = _make_array()
        result = missing_prep_transform(arr.copy())
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, arr.shape)

    def test_standardize_array(self):
        arr = _make_array(n=500)
        result = missing_prep_transform(arr.copy())
        np.testing.assert_allclose(result[:, 0].mean(), 0, atol=1e-10)
        np.testing.assert_allclose(result[:, 0].std(ddof=0), 1, atol=1e-10)

    def test_normalize_array(self):
        arr = _make_array(n=200)
        result = missing_prep_transform(arr.copy(), continuous_transform='normalize')
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_train_test_arrays(self):
        arr = _make_array(n=100)
        train, test = arr[:80], arr[80:]
        train_out, test_out = missing_prep_transform(train.copy(), test_data=test.copy())
        self.assertIsInstance(train_out, np.ndarray)
        self.assertIsInstance(test_out, np.ndarray)
        self.assertEqual(train_out.shape[1], test_out.shape[1])

    def test_does_not_mutate_input_array(self):
        arr = _make_array()
        original = arr.copy()
        missing_prep_transform(arr)
        np.testing.assert_array_equal(arr, original)


class TestRemoveMCARArray(unittest.TestCase):

    def test_returns_ndarray(self):
        arr = _make_array(n=200)
        result = remove_mcar(arr, pct=0.20, random_state=0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, arr.shape)

    def test_introduces_nan(self):
        arr = _make_array(n=500)
        result = remove_mcar(arr, pct=0.20, random_state=0)
        self.assertGreater(np.isnan(result).sum(), 0)

    def test_approximate_missing_rate_array(self):
        arr = _make_array(n=2000)
        pct = 0.15
        result = remove_mcar(arr, pct=pct, random_state=0)
        actual = np.isnan(result).sum() / result.size
        self.assertAlmostEqual(actual, pct, delta=0.03)

    def test_does_not_mutate_input_array(self):
        arr = _make_array()
        original = arr.copy()
        remove_mcar(arr, pct=0.10, random_state=0)
        np.testing.assert_array_equal(arr, original)

    def test_reproducible_array(self):
        arr = _make_array(n=200)
        r1 = remove_mcar(arr.copy(), pct=0.10, random_state=7)
        r2 = remove_mcar(arr.copy(), pct=0.10, random_state=7)
        np.testing.assert_array_equal(np.isnan(r1), np.isnan(r2))


class TestRemoveMNARArray(unittest.TestCase):

    def test_returns_ndarray(self):
        arr = _make_array(n=200)
        result = remove_mnar(arr, q=0.8, p=0.9, random_state=0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, arr.shape)

    def test_introduces_nan(self):
        arr = _make_array(n=500)
        result = remove_mnar(arr, q=0.8, p=0.9, random_state=0)
        self.assertGreater(np.isnan(result).sum(), 0)

    def test_missing_concentrated_above_quantile_array(self):
        arr = _make_array(n=2000)
        result = remove_mnar(arr, q=0.8, p=1.0, columns=[0], random_state=0)
        threshold = np.quantile(arr[:, 0], 0.8)
        missing_rows = np.isnan(result[:, 0])
        self.assertTrue((arr[missing_rows, 0] > threshold).all())

    def test_subset_columns_array(self):
        arr = _make_array(n=500)
        result = remove_mnar(arr, q=0.8, p=1.0, columns=[0], random_state=0)
        self.assertGreater(np.isnan(result[:, 0]).sum(), 0)
        self.assertFalse(np.isnan(result[:, 1]).any())
        self.assertFalse(np.isnan(result[:, 2]).any())

    def test_does_not_mutate_input_array(self):
        arr = _make_array()
        original = arr.copy()
        remove_mnar(arr, q=0.7, p=0.5, random_state=0)
        np.testing.assert_array_equal(arr, original)

    def test_unknown_column_index_raises(self):
        arr = _make_array(n=50)
        with self.assertRaises((ValueError, KeyError)):
            remove_mnar(arr, q=0.8, p=0.5, columns=[99])


class TestRemoveMARArray(unittest.TestCase):

    def test_returns_ndarray(self):
        arr = _make_array(n=200)
        result = remove_mar(arr, q=0.5, p=0.8, targets=[0], conditionals=[1], random_state=0)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, arr.shape)

    def test_introduces_nan(self):
        arr = _make_array(n=500)
        result = remove_mar(arr, q=0.5, p=0.8, targets=[0], conditionals=[1], random_state=0)
        self.assertGreater(np.isnan(result[:, 0]).sum(), 0)

    def test_conditional_column_unchanged_array(self):
        arr = _make_array(n=500)
        result = remove_mar(arr, q=0.5, p=0.9, targets=[0], conditionals=[1], random_state=0)
        self.assertFalse(np.isnan(result[:, 1]).any())

    def test_does_not_mutate_input_array(self):
        arr = _make_array()
        original = arr.copy()
        remove_mar(arr, q=0.5, p=0.5, targets=[0], conditionals=[1], random_state=0)
        np.testing.assert_array_equal(arr, original)


if __name__ == "__main__":
    unittest.main()

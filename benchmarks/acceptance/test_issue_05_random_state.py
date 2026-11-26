from pandas.testing import assert_frame_equal

from dataset import make_dataset


def test_issue_05_random_state():
    """random_state должен обеспечивать воспроизводимость и менять данные."""

    first = make_dataset(n_samples=100, random_state=17)
    repeated = make_dataset(n_samples=100, random_state=17)
    different_seed = make_dataset(n_samples=100, random_state=19)

    assert_frame_equal(first, repeated)
    assert not first.equals(different_seed)

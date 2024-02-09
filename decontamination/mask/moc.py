# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

########################################################################################################################

def order_index_to_nuniq(orders: np.ndarray, indices: np.ndarray) -> np.ndarray:

    """
    Encodes HEALPix orders and pixel indices to unique identifiers (nuniq).

    Parameters
    ----------
    orders : np.ndarray
        Array of HEALPix orders.
    indices : np.ndarray
        Array of HEALPix pixel indices.

    Returns
    -------
    np.ndarray
        Array of nuniq values for the given orders and pixel indices combinations.
    """

    return 4 * (4 ** orders) + indices

########################################################################################################################

def nuniq_to_order_index(nuniqs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

    """
    Decodes HEALPix nuniq identifiers to their corresponding orders and pixel indices.

    Parameters
    ----------
    nuniqs : np.ndarray
        Array of HEALPix nuniq identifiers.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        First array represents the HEALPix orders for the given nuniq values.
        Second array represents the HEALPix pixel indices for the given nuniq values.
    """

    orders = np.floor_divide(np.log2(np.floor_divide(nuniqs, 4)), 2).astype(nuniqs.dtype)

    indices = nuniqs - 4 * (4 ** orders)

    return orders, indices

########################################################################################################################

def moc_to_healpix(moc_orders: np.ndarray, moc_indices: np.ndarray, order_new: int) -> np.ndarray:

    """
    Refines a given Multi-Order Coverage (MOC) to a specified order.

    Parameters
    ----------
    moc_orders : np.ndarray
        Array of HEALPix orders for each MOC cell.
    moc_indices : np.ndarray
        Array of HEALPix pixel indices for each MOC cell.
    order_new : int
        The desired refinement order.

    Returns
    -------
    np.ndarray
        Array of HEALPix pixel indices at the new order.
    """

    order_new = int(order_new)

    ####################################################################################################################

    if moc_orders.shape != moc_indices.shape:
        raise ValueError('Both `moc_orders` and `moc_indices` must have the same shape')

    if np.max(moc_orders) > order_new:
        raise ValueError('The refinement level must be higher than the MOC smallest healpix order')

    ####################################################################################################################

    # A HEALPix pixel at nside1 can be decomposed into their children (at nside2 > nside1)
    # following the hierarchical property. Using NESTED numbering scheme -- what must be
    # the case when working with MOCs -- the children of a pixel p have indexes
    # {4p, 4p+1, 4p+2, 4p+3}.
    #
    #          /\
    #         /  \
    #        /    \
    #       /      \
    #      /        \
    #     /\  4p+3  /\
    #    /  \      /  \
    #   /    \    /    \
    #  /      \  /      \
    # / 4p+2   \/  4p+1  \
    # \        /\        /
    #  \      /p \      /
    #   \    /    \    /
    #    \  /      \  /
    #     \/        \/
    #      \  4p+0  /
    #       \      /
    #        \    /
    #         \  /
    #          \/

    result = []

    for order, index in zip(moc_orders, moc_indices):

        factor = 4 ** (order_new - order)

        base_index = index * factor

        result.extend(range(base_index, base_index + factor))

    ####################################################################################################################

    return np.array(result, dtype = moc_indices.dtype)

########################################################################################################################

def wmoc_to_healpix(moc_orders: np.ndarray, moc_indices: np.ndarray, moc_weights: np.ndarray, order_new: int) -> typing.Tuple[np.ndarray, np.ndarray]:

    """
    Refines a given Weighted Multi-Order Coverage (WMOC) to a specified order, returning the HEALPix indices at the new order and the associated weights.

    Parameters
    ----------
    moc_orders : np.ndarray
        Array of HEALPix orders for each MOC cell.
    moc_indices : np.ndarray
        Array of HEALPix pixel indices for each MOC cell.
    moc_weights : np.ndarray
        Array of associated values/weights for each MOC cell.
    order_new : int
        The desired refinement order.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        First array contains the HEALPix pixel indices at the new order.
        Second array contains the associated values/weights at the new order.
    """

    order_new = int(order_new)

    ####################################################################################################################

    if moc_orders.shape != moc_indices.shape != moc_weights.shape:
        raise ValueError('Both `moc_orders`, `moc_indices` and `moc_weights` must have the same shape')

    if np.max(moc_orders) > order_new:
        raise ValueError('The refinement level must be higher than the WMOC smallest healpix order')

    ####################################################################################################################

    # A HEALPix pixel at nside1 can be decomposed into their children (at nside2 > nside1)
    # following the hierarchical property. Using NESTED numbering scheme -- what must be
    # the case when working with MOCs -- the children of a pixel p have indexes
    # {4p, 4p+1, 4p+2, 4p+3}.
    #
    #          /\
    #         /  \
    #        /    \
    #       /      \
    #      /        \
    #     /\  4p+3  /\
    #    /  \      /  \
    #   /    \    /    \
    #  /      \  /      \
    # / 4p+2   \/  4p+1  \
    # \        /\        /
    #  \      /p \      /
    #   \    /    \    /
    #    \  /      \  /
    #     \/        \/
    #      \  4p+0  /
    #       \      /
    #        \    /
    #         \  /
    #          \/

    result_indices = []
    result_weights = []

    for order, index, weight in zip(moc_orders, moc_indices, moc_weights):

        factor = 4 ** (order_new - order)

        base_index = index * factor

        result_indices.extend(range(base_index, base_index + factor))
        result_weights.extend(          [weight] * factor           )

    ####################################################################################################################

    return (
        np.array(result_indices, dtype = np.int64),
        np.array(result_weights, dtype = moc_weights.dtype),
    )

########################################################################################################################

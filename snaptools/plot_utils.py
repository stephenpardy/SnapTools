import numpy as np

def delete_overlapping_xlabels(fig, ax):

    ''' Deletes overlapping xtick labels'''


    fig.canvas.draw()
    major_labels = ax.get_xticklabels(minor=False)
    new_major_text = [item.get_text() for item in major_labels]
    bboxes = [label.get_window_extent() for label in major_labels]

    bbox_overlaps = check_overlaps(bboxes)
    any_overlaps = any(bbox_overlaps)

    while any_overlaps:
        i = np.argmax(bbox_overlaps)
        new_major_text[i] = ''
        bboxes[i].set_points(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
        bbox_overlaps = check_overlaps(bboxes)
        any_overlaps = any(bbox_overlaps)

    minor_labels = ax.get_xticklabels(minor=True)
    new_minor_text = [item.get_text() for item in minor_labels]
    minor_bboxes = [label.get_window_extent() for label in minor_labels]

    minor_bbox_overlaps = check_overlaps_minor(minor_bboxes, bboxes)
    any_minor_overlaps = any(minor_bbox_overlaps)
    while any_minor_overlaps:
        i = np.argmax(minor_bbox_overlaps)
        new_minor_text[i] = ''
        minor_bboxes[i].set_points(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
        minor_bbox_overlaps = check_overlaps_minor(minor_bboxes, bboxes)
        any_minor_overlaps = any(minor_bbox_overlaps)


    minor_bbox_overlaps = check_overlaps(minor_bboxes)
    any_minor_overlaps = any(minor_bbox_overlaps)
    while any_minor_overlaps:
        i = np.argmax(minor_bbox_overlaps)
        new_minor_text[i] = ''
        minor_bboxes[i].set_points(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
        minor_bbox_overlaps = check_overlaps(minor_bboxes)
        any_minor_overlaps = any(minor_bbox_overlaps)

    ax.set_xticklabels(new_major_text)
    ax.set_xticklabels(new_minor_text, minor=True)


def check_overlaps(bboxes):

    '''
    takes list of bboxes
    returns a list of how many times each bbox overlaps with other bboxes
    '''

    num_overlaps = [0] * len(bboxes)
    for i, box in enumerate(bboxes):
        for other_box in bboxes:
            if (box != other_box):
                num_overlaps[i] += overlaps(box, other_box)

    return num_overlaps


def check_overlaps_minor(bboxes, other_bboxes):

    '''
    takes two lists of bboxes
    returns a list of how many times each bbox overlaps with other bboxes in each list
    '''

    num_overlaps = [0] * len(bboxes)
    for i, box in enumerate(bboxes):
        for other_box in bboxes:
            if (box != other_box):
                num_overlaps[i] += overlaps(box, other_box)
        for other_box in other_bboxes:
            if (box != other_box):
                num_overlaps[i] += overlaps(box, other_box)

    return num_overlaps


def overlaps(box, other):
        """
        Returns True if this bounding box overlaps with the given
        bounding box *other*.
        """
        ax1, ay1, ax2, ay2 = box._get_extents()
        bx1, by1, bx2, by2 = other._get_extents()
        if any(np.isnan(v) for v in [ax1, ay1, ax2, ay2, bx1, by1, bx2, by2]):
            return False

        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2

        return not ((bx2 < ax1) or
                    (by2 < ay1) or
                    (bx1 > ax2) or
                    (by1 > ay2))
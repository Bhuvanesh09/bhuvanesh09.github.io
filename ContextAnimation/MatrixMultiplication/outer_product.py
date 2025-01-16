from manim import *

BACKGROUND_COLOR = "#FAF8F1"
color_scheme = ["#4A90C0", "#FFA54F", "#9370DB", "#66CDAA", "#CD5C5C"]
THEME_COLORS = [ManimColor(color) for color in color_scheme]
FW, FH = 12, 10
HIGHLIGHT_GAP = 0.05
HIGHLIGHT_STROKE_WIDTH = 6


def coor(x, y):
    return (x - FW / 2, y - FH / 2, 0)


class Matrix:
    def __init__(
        self,
        shape=(2, 2),
        gap_between_cells=0.1,
        side_length=1,
        edge_color=THEME_COLORS[0],
    ):
        self.gap_between_cells = gap_between_cells
        self.shape = shape
        self.side_length = side_length
        self.edge_color = edge_color

    def fill_water_in_box(self, i, j, opacity):
        return self.objects[i][j].animate.set_fill(
            color=self.edge_color, opacity=opacity
        )

    def make_highlight_row(self, highlight_color=THEME_COLORS[3]):
        row_to_highlight = self.objects[0]

        highlight_row = Rectangle(
            height=row_to_highlight.height + HIGHLIGHT_GAP,
            width=row_to_highlight.width + HIGHLIGHT_GAP,
            stroke_color=highlight_color,
            stroke_width=HIGHLIGHT_STROKE_WIDTH,
        )
        highlight_row.move_to(row_to_highlight.get_center())
        self.highlight_row = highlight_row
        return highlight_row

    def teleport_highlight_row(self, i):
        row_to_highlight = self.objects[i]
        return self.highlight_row.move_to(row_to_highlight.get_center())

    def move_highlight_row(self, i):
        row_to_highlight = self.objects[i]
        return self.highlight_row.animate.move_to(row_to_highlight.get_center())

    def make_highlight_column(self, highlight_color=THEME_COLORS[3]):
        col_to_highlight = VGroup(*[row[0] for row in self.objects])

        highlight_col = Rectangle(
            height=col_to_highlight.height + HIGHLIGHT_GAP,
            width=col_to_highlight.width + HIGHLIGHT_GAP,
            stroke_color=highlight_color,
            stroke_width=HIGHLIGHT_STROKE_WIDTH,
        )
        highlight_col.move_to(col_to_highlight.get_center_of_mass())
        self.highlight_col = highlight_col
        return highlight_col

    def teleport_highlight_column(self, j):
        col_to_highlight = VGroup(*[row[j] for row in self.objects])
        return self.highlight_col.move_to(col_to_highlight.get_center())

    def move_highlight_column(self, j):
        col_to_highlight = VGroup(*[row[j] for row in self.objects])
        return self.highlight_col.animate.move_to(col_to_highlight.get_center())

    def make_highlight_square(self, highlight_color=THEME_COLORS[2]):
        square_to_highlight = self.objects[0][0]
        highlight_square = Square(
            side_length=square_to_highlight.side_length - HIGHLIGHT_GAP,
            stroke_color=highlight_color,
            stroke_width=HIGHLIGHT_STROKE_WIDTH,
            fill_opacity=0,
        )
        highlight_square.move_to(square_to_highlight.get_center())
        self.highlight_square = highlight_square
        return highlight_square

    def teleport_highlight_square(self, i, j):
        square_to_highlight = self.objects[i][j]
        return self.highlight_square.move_to(square_to_highlight.get_center())

    def move_highlight_square(self, i, j):
        square_to_highlight = self.objects[i][j]
        return self.highlight_square.animate.move_to(square_to_highlight.get_center())

    def make_objects(self):
        rows, cols = self.shape
        matrix_mob = VGroup()
        for i in range(rows):
            row_mob = VGroup()
            for j in range(cols):
                cell = Square(
                    side_length=self.side_length,
                    stroke_color=self.edge_color,
                    fill_opacity=0,
                )
                row_mob.add(cell)
            row_mob.arrange(RIGHT, buff=self.gap_between_cells)
            matrix_mob.add(row_mob)
        matrix_mob.arrange(DOWN, buff=self.gap_between_cells)
        self.objects = matrix_mob
        return matrix_mob


class VariableTracker:
    def __init__(self, variable, value, i, j, color=THEME_COLORS[0], scale=1.5):
        self.variable = variable
        self.value = value
        self.color = color

        self.objects = VGroup()
        static_text = Tex(variable)
        static_text.set_color(color)
        static_text.scale(scale)
        self.objects.add(static_text)
        self.number_tracker = ValueTracker(value)
        self.number_display = Integer(self.number_tracker.get_value())
        self.number_display.set_color(self.color)
        self.number_display.scale(scale)
        self.number_display.add_updater(
            lambda d: d.set_value(self.number_tracker.get_value())
        )
        self.objects.add(self.number_display)
        self.objects.arrange(RIGHT, buff=0.2)
        self.objects.move_to(coor(i, j))

    def get_objects(self):
        return self.objects

    def update_value(self, value):
        return self.number_tracker.animate.set_value(value)


class MatrixMultiplicationOuter(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        self.camera.frame_width = 12
        self.camera.frame_height = 10
        self.gap_between_cells = 0.0
        m1 = Matrix(
            (3, 4), gap_between_cells=self.gap_between_cells, edge_color=THEME_COLORS[0]
        )
        m2 = Matrix(
            (4, 5), gap_between_cells=self.gap_between_cells, edge_color=THEME_COLORS[4]
        )
        m3 = Matrix(
            (3, 5), gap_between_cells=self.gap_between_cells, edge_color=THEME_COLORS[2]
        )
        m1.make_objects()
        m2.make_objects()
        m3.make_objects()
        m1.objects.move_to(coor(3, 2.5))
        m2.objects.move_to(coor(8.5, 1 + 3 + 1 + 2))
        m3.objects.move_to(coor(8.5, 2.5))
        self.add(m1.objects, m2.objects, m3.objects)

        var_i = VariableTracker(
            variable=r"$i=$", value=0, i=3, j=8, color=THEME_COLORS[3]
        )
        var_j = VariableTracker(
            variable=r"$j=$", value=0, i=3, j=7, color=THEME_COLORS[1]
        )
        var_k = VariableTracker(
            variable=r"$k=$", value=0, i=3, j=6, color=THEME_COLORS[2]
        )

        self.add(
            m1.make_highlight_column(highlight_color=THEME_COLORS[2]),
            m2.make_highlight_row(highlight_color=THEME_COLORS[2]),
            m3.make_highlight_column(highlight_color=THEME_COLORS[1]),
            m3.make_highlight_row(highlight_color=THEME_COLORS[3]),
            m3.make_highlight_square(highlight_color=THEME_COLORS[2]),
            m1.make_highlight_square(highlight_color=THEME_COLORS[0]),
            m2.make_highlight_square(highlight_color=THEME_COLORS[4]),
            var_i.get_objects(),
            var_j.get_objects(),
            var_k.get_objects(),
        )
        for k in range(m2.shape[0]):
            animation_list = []
            animation_list.append(m2.move_highlight_row(k))
            animation_list.append(m1.move_highlight_column(k))
            animation_list.append(var_k.update_value(k))
            for i in range(m1.shape[0]):
                for j in range(m2.shape[1]):
                    animation_list.append(m3.move_highlight_column(j))
                    animation_list.append(m3.move_highlight_row(i))
                    animation_list.append(m1.move_highlight_square(i, k))
                    animation_list.append(m2.move_highlight_square(k, j))
                    animation_list.append(m3.move_highlight_square(i, j))
                    animation_list.append(
                        m3.fill_water_in_box(i, j, float((k + 1) / (m2.shape[0] + 1)))
                    )
                    animation_list.append(var_i.update_value(i))
                    animation_list.append(var_j.update_value(j))
                    self.play(*animation_list, run_time=0.25)
                    self.wait(0.1)

        self.wait(1)

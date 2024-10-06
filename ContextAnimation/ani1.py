from manim import *

class TokenGeneration(Scene):
    def construct(self):
        # Set the background color
        self.camera.background_color = "#FAF8F2"  # Using the extracted color

        # List of tokens to visualize
        tokens = [f"T{i+1}" for i in range(5)]  # Generates ["T1", "T2", ..., "T5"]

        # Create transparent boxes with the specified border color
        token_boxes = [Square(color="#171717", fill_opacity=0, stroke_width=2).scale(0.2) for _ in tokens]
        token_texts = [Text(token, font_size=24, color="#171717") for token in tokens]

        # Add margin between boxes
        margin = 0.1  # Adjust this value for more or less spacing

        # Position the boxes in a line with margins
        for i, box in enumerate(token_boxes):
            box.move_to(RIGHT * i * (1 + margin))  # Multiply by (1 + margin) to add spacing
            token_texts[i].move_to(box.get_center())

        # Show tokens one by one to illustrate generation
        for i in range(len(tokens)):
            self.play(Create(token_boxes[i]), Write(token_texts[i]))
            self.wait(0.5)

            if i > 0:
                # Draw all curved arrows for the current token at once
                arcs = VGroup(*[
                    ArcBetweenPoints(
                        start=token_boxes[i].get_top(),
                        end=token_boxes[j].get_top(),
                        angle=PI/3,  # Angle for an upward curve
                        color="#171717"
                    ) for j in range(i)
                ])
                # Play creation of all arcs at once
                self.play(*[Create(arc) for arc in arcs])
                self.wait(0.2)
                self.play(*[FadeOut(arc) for arc in arcs])

        # Highlight the final token to show attention spreading to all previous tokens
        for i in range(len(tokens)):
            highlight = SurroundingRectangle(token_boxes[i], color=YELLOW, buff=0.1)
            self.play(Create(highlight))
            self.wait(0.3)
            # Use FadeOut instead of Uncreate to fade out the highlight
            self.play(FadeOut(highlight))


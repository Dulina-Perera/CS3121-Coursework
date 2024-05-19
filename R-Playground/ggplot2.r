library(tidyverse)

# Load the mpg dataset.
data(mpg, package = "ggplot2")
print(mpg)

ggplot2::ggplot(data = mpg) +
  ggplot2::geom_point(mapping = ggplot2::aes(x = displ, y = hwy, colour = class))

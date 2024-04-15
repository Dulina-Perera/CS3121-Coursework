library(tidyverse)

mpg

ggplot2::ggplot(data = mpg) +
  ggplot2::geom_point(mapping = ggplot2::aes(x = displ, y = hwy, color = class))

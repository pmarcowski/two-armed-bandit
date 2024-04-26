# Title: Two-armed bandit game with reinforcement learning analysis Shiny app
# Author: Przemyslaw Marcowski, PhD
# Email: p.marcowski@gmail.com
# Date: 2024-02-03
# Copyright (c) 2024 Przemyslaw Marcowski

# This code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script implements a Two-Armed Bandit game using Shiny, where players
# interact with a bandit, choosing between two arms that offer different
# reward potentials. The goal is to maximize the total score by deducing
# which arm currently offers the higher bonus. The game includes multiple
# rounds, allowing players to refine their strategies.
#
# After completing all rounds, a reinforcement learning model (Kalman UCB)
# analyzes the player's choices, and the decision history characteristics
# are displayed using visualizations. The results include:
#
# - Arm Selection Distribution: Frequency of selecting each arm throughout
#   the game.
# - UCB Value for Each Arm: Upper Confidence Bound (UCB) value estimated by
#   the model for each arm over time.
# - Probability of Choosing Each Arm: Probability of selecting each arm
#   based on the model's estimates.
# - Reaction Times: Player's reaction times for each decision throughout
#   the game.
# - Cumulative Reward: Accumulated rewards over the course of the game.
# - Cumulative Regret: Difference between the maximum possible reward and
#   the player's actual reward.

# Global ------------------------------------------------------------------

# Load packages
library(shiny)
library(shinyBS)
library(tidyverse)
library(patchwork)

# Initialize parameters
num_trials <- 50
flip_prob <- 0.075
payoff_range <- c(10, 40)
flip_bonus <- 20
m <- "kalman_ucb"

## Model functions ----

get_fit <- function(dfit, m, lim, nfit = 1, ...) {
  # Fits model using L-BFGS-B optimizer.
  # Fitting is performed by MINIMIZATION.
  # Fitting is repeated until nfit successful fits are obtained.
  #
  # Args:
  #   dfit: Data for fitting.
  #   m: Model string for model function to be fitted
  #   lim: Matrix with lower and upper limits per parameter.
  #   nfit: Number of fits performed to get best fit
  #   ...: Reserved
  #
  # Returns:
  #   Estimator value resulting from best fit.
  #   Parameter values resulting from best fit.

  .rand_starts <- function(m) {
    # Randomizes start values for each parameter within limits
    #
    # Args:
    #   x: Model string for which parameters are randomized
    #
    # Returns:
    #   Randomized starting value per parameter
    start <- c()
    for (i in 1:ncol(lim)) {
      start[i] <- runif(
        1,
        ifelse(lim[1, i] < 1e-5, 1e-5, lim[1, i]),
        ifelse(lim[2, i] > 1e+2, 1e+2, lim[2, i])
      )
    }
    start
  }

  fit_v <- Inf
  fit_c <- 0
  while (fit_c < nfit) {
    res_tmp <- try(if (ncol(lim) > 1) {
      optim(
        .rand_starts(m), get(m),
        d = dfit, ...,
        method = "L-BFGS-B",
        lower = lim[1, ], upper = lim[2, ]
      )
    }, silent = FALSE)
    if (class(res_tmp) != "try-error") {
      fit_c <- fit_c + 1
      if (res_tmp$value < fit_v) {
        fit_v <- res_tmp$value
        fit_p <- res_tmp$par
      }
    } else {
      print("Error occurred while fitting model.")
    }
  }
  if (fit_v != -Inf) {
    return(c(fit_v, fit_p))
  } else {
    return(NULL)
  }
}

# Create matrix with parameter limits for kalman_ucb model
lim <- matrix(c(
  1e-5, 1e+2, # process_noise
  1e-5, 1e+2, # obs_noise
  0.1, 10 # ucb_coef
), nrow = 2)

kalman_ucb <- function(params, data, returns = c("nlp", "value"), epsilon = 1e-3) {
  # Kalman filter with UCB rule for a two-armed bandit.
  #
  # Args:
  #   params: Parameter vector (process_noise, obs_noise, ucb_coef)
  #   data: Data with 'choice' and 'outcome' columns
  #   returns: Specifies the return type (default: nlp)
  #   epsilon: Smoothing factor for probabilities (default: 1e-3)
  #
  # Returns:
  #   Depending on the 'returns' parameter, returns a data frame with action
  #   values, probabilities, and prediction error or negative log posterior.

  choice <- data$choice # 1 for left, 2 for right arm
  outcome <- data$payout # points received per trial
  num_trials <- length(choice) # number of trials

  # Initialize Kalman filter parameters
  process_noise <- params[1]
  obs_noise <- params[2]
  ucb_coef <- params[3]

  # Initialize arrays to store values
  reward_estimate <- matrix(0, ncol = 2, nrow = num_trials + 1)
  error_cov <- matrix(1, ncol = 2, nrow = num_trials + 1)
  kalman_gain <- matrix(0, ncol = 2, nrow = num_trials)
  innovation <- matrix(0, ncol = 2, nrow = num_trials)
  selections <- numeric(2) # to track number of times each arm is selected

  for (t in 1:num_trials) {
    # Update arm selections
    selections[choice[t]] <- selections[choice[t]] + 1

    # Prediction step
    predicted_cov <- error_cov[t, ] + process_noise

    # Update step
    kalman_gain[t, ] <- predicted_cov / (predicted_cov + obs_noise)
    innovation[t, choice[t]] <- outcome[t] - reward_estimate[t, choice[t]]
    reward_estimate[t + 1, ] <- reward_estimate[t, ] + kalman_gain[t, ] * innovation[t, ]
    error_cov[t + 1, ] <- (1 - kalman_gain[t, ]) * predicted_cov
  }

  # Calculate UCB values
  ucb_values <- matrix(0, ncol = 2, nrow = num_trials)
  for (t in 1:num_trials) {
    ucb_values[t, ] <- reward_estimate[t, ] + ucb_coef * sqrt(log(t) / ifelse(selections == 0, 1, selections))
  }

  # Calculate choice probabilities using UCB
  choice_prob <- exp(ucb_values) / rowSums(exp(ucb_values))
  choice_prob <- epsilon * 0.5 + (1 - epsilon) * choice_prob

  # Calculate log-likelihood
  log_lik <- sum(log(choice_prob[cbind(1:num_trials, choice)]))

  # Calculate log prior probability of the parameters
  log_prior <- sum(dnorm(params, mean = 0, sd = 10, log = TRUE))

  # Calculate negative log posterior
  neg_log_posterior <- -log_lik - log_prior

  if (returns[1] == "value") {
    return(data.frame(
      trial = 1:num_trials,
      choice = choice,
      prob = choice_prob,
      reward = reward_estimate[-1, ], # exclude initial value
      ucb = ucb_values # exclude initial value
    ))
  }

  if (returns[1] == "nlp") {
    return(neg_log_posterior)
  }
}

cross_entropy <- function(y, yhat) {
  # Computes cross-entropy between true choices and predicted choice probabilities
  #
  # Args:
  #   y: Integer. Vector containing options chosen per trial.
  #   yhat: Numeric. Predicted probabilities of choosing each option per trial.
  #
  # Returns:
  #   Numeric. Trial-wise cross-entropy values
  nt <- length(y)
  if (!nt == nrow(yhat)) stop("Truth and predictions are of different length")
  ce <- rep(0, nt)
  for (i in 1:nt) ce[i] <- -log(yhat[i, y[i]])
  return(ce)
}

## Helper functions ----

bandit_image <- function(arm) {
  # Generates a path to the bandit arm image based on the selected arm.
  # This function updates the UI to reflect which arm of the bandit is active.
  #
  # Args:
  #   arm: Integer. 1 for left arm, 2 for right arm, others for default image.
  #
  # Returns:
  #   A string path to the appropriate bandit arm image.

  if (arm == 1) {
    "./www/arm-left.png"
  } else if (arm == 2) {
    "./www/arm-right.png"
  } else {
    "./www/bandit.png"
  }
}

# UI ----------------------------------------------------------------------

ui <- fluidPage(
  title = "Two-Armed Bandit Game",

  # CSS styling for layout
  tags$head(
    tags$script(HTML('
    Shiny.addCustomMessageHandler("resetGIF", function(message) {
      var gif = document.getElementById("payoff_gif");
      gif.style.visibility = "visible";
      gif.src = "";
      gif.offsetHeight;
      gif.src = message.src;
      setTimeout(function() {
        gif.style.visibility = "hidden";
      }, 1000);
    });
  ')),
    tags$style(HTML("
      body {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        flex-direction: column;
        font-size: 1.5rem;
      }
      .centered-content {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
      }
      .game-container {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 1rem;
      }
      .pull-arm {
        margin-top: 1rem;
        padding: 1rem;
        font-size: 1rem;
      }
      .outcome-payoff-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 1rem;
      }
      .help-button {
        position: absolute;
        top: 1rem;
        left: 50%;
        transform: translateX(-50%);
        border-radius: 100%;
        width: 3rem;
        height: 3rem;
        font-size: 2rem;
        padding: 0;
      }
      .bandit-and-buttons {
          margin: 2rem;
      }
      @media screen and (max-width: 600px) {
        .centered-content {
          padding: 1rem;
        }
        .pull-arm {
          font-size: 1em;
        }
      }
    "))
  ),

  # Window ----

  # Modal for instructions
  bsModal(
    "modal_instructions", "Author", "help_button_about",
    size = "large",
    p(
      "Welcome to the Two-Armed Bandit Game! I'm Przemek, a San Diego-based
      researcher and data scientist with a passion for using data to make things
      more interesting. I built this app as a fun way to explore how we learn
      and adapt our strategies when making decisions."
    ),
    p(HTML("You can explore my work <a href='https://przemyslawmarcowski.com' target='_blank'>here</a>."))
  ),

  # Modal for game in progress
  bsModal(
    "modal_in_progress", "Game Rules", "help_button_in_progress",
    size = "large",
    p(
      "In this game, you are presented with a two-armed bandit. Each arm offers
      a different reward potential. Your goal is to maximize your total score
      by selecting the arm that currently offers the higher bonus. The game
      consists of multiple rounds, allowing you to refine your strategy as
      you play."
    )
  ),

  # Modal for results
  bsModal(
    "modal_results", "Results Explanation", "help_button_results",
    size = "large",
    p(
      "The results displayed here are based on your choices throughout the
      game. A reinforcement learning model called Kalman UCB was used to
      analyze your decision-making process. The model estimates the expected
      reward and uncertainty for each arm and explains the exploration-
      exploitation trade-off."
    ),
    p("The plots show various aspects of your performance:"),
    tags$ul(
      tags$li(
        "Arm Selection Distribution: Displays the frequency of
        selecting each arm throughout the game. It helps understand your
        preference for each arm."
      ),
      tags$li(
        "UCB Value for Each Arm: Shows the Upper Confidence
        Bound (UCB) value estimated by the model for each arm over time.
        The UCB value balances the expected reward and uncertainty."
      ),
      tags$li(
        "Probability of Choosing Each Arm: Illustrates the
        probability of selecting each arm based on the model's estimates.
        It reflects your likelihood of exploring or exploiting each arm."
      ),
      tags$li(
        "Reaction Times: Presents your reaction times for each
        decision throughout the game. It can reveal patterns or changes
        in your decision-making speed."
      ),
      tags$li(
        "Cumulative Reward: Shows the accumulated rewards over
        the course of the game. It measures your overall performance and
        how well you maximized your total score."
      ),
      tags$li(
        "Cumulative Regret: Depicts the difference between the
        maximum possible reward and your actual reward. It quantifies
        the potential loss due to suboptimal choices."
      )
    ),
    p("The vertical dotted lines indicate trials where the higher reward
      potential switched to the other arm. The shaded areas indicate which
      arm was selected in those trials.")
  ),
  div(
    class = "centered-content",

    # Show instructions screen
    conditionalPanel(
      condition = "output.game_stage == 'instructions'",
      actionButton("help_button_about", "?", class = "btn-info help-button", style = "color: #000000; background-color: #FFFFFF; border-color: #000000"),
      h2("Instructions"),
      p("Welcome to the Two-Armed Bandit Game!"),
      p(HTML("In this game, you'll interact with a two-armed bandit, each arm offering <br>
             a different reward potential. Try to deduce which arm currently offers <br>
             the higher bonus to optimize your score. Your goal is to maximize your total score.")),
      p(HTML("The game includes multiple rounds, allowing you to refine your strategy as you play. <br>
             After completing all the rounds, a reinforcement learning model will analyze your choices <br>
             and your decision history will be displayed, providing insights into your strategic patterns.")),
      p(HTML("You can click the <b>?</b> icon at the top of the screen anytime <br>
             to access detailed information and helpful tips.")),
      p(HTML("Click the 'Start Game' button to begin. Good luck!")),
      br(),
      actionButton("start_button", "Start Game", class = "btn-primary")
    ),

    # Play game loop
    conditionalPanel(
      condition = "output.game_stage == 'in_progress'",
      div(
        class = "game-container",
        actionButton("help_button_in_progress", "?", class = "btn-info help-button", style = "color: #000000; background-color: #FFFFFF; border-color: #000000"),
        h3(textOutput("trial_num")),
        div(
          class = "bandit-and-buttons",
          imageOutput("bandit_image", height = "auto", width = "auto"),
          actionButton("left_arm", "Pull Left", class = "btn-secondary pull-arm"),
          actionButton("right_arm", "Pull Right", class = "btn-secondary pull-arm")
        ),
        div(
          class = "outcome-payoff-container",
          uiOutput("outcome_animation"),
          h3(class = "payoff-text", textOutput("payoff")),
          h4(class = "total-points", textOutput("total_points"))
        )
      )
    ),

    # Show game over screen
    conditionalPanel(
      condition = "output.game_stage == 'game_over'",
      div(
        class = "centered-content",
        h2("Game Over"),
        p("Congratulations! You have completed the game."),
        h4(textOutput("final_score")),
        p("Your choices will now be analyzed by a reinforcement learning model."),
        p("Different characteristics of your decision making will be displayed."),
        br(),
        actionButton("continue_button", "Continue", class = "btn-primary")
      )
    ),

    # Show results screen
    conditionalPanel(
      condition = "output.game_stage == 'results'",
      div(
        class = "centered-content",
        actionButton("help_button_results", "?", class = "btn-info help-button", style = "color: #000000; background-color: #FFFFFF; border-color: #000000"),
        h2("Results"),
        plotOutput("results_fig", width = "auto", height = "auto", inline = TRUE),
        br(),
        fluidRow(
          column(6, align = "right", actionButton("end_button", "End", class = "btn-primary")),
          column(6, align = "left", downloadButton("download_data", "Download Data"))
        )
      )
    ),

    # Show end screen
    conditionalPanel(
      condition = "output.game_stage == 'end'",
      div(
        class = "centered-content",
        h2("Thank you for playing!"),
        p("You can now close this window to exit or refresh the page to play again.")
      )
    )
  )
)

# Server ------------------------------------------------------------------

server <- function(input, output, session) {
  .arm_pull <- function(arm) {
    # Handles arm pulls
    current_time <- Sys.time()
    reaction_time <- as.numeric(difftime(current_time, previous_pull_time(), units = "secs"))
    reaction_times(c(reaction_times(), reaction_time))
    previous_pull_time(current_time)
    choices(c(choices(), ifelse(arm == 1, 1, 2)))
    payoff_value <- payoffs()[[arm]]
    payouts(c(payouts(), payoff_value))
    trial(trial() + 1)
    selected_arm(arm)
  }

  ## Reactive values ----

  trial <- reactiveVal(1)
  reaction_times <- reactiveVal(numeric(0))
  previous_pull_time <- reactiveVal(NULL)
  choices <- reactiveVal(numeric(0))
  payouts <- reactiveVal(numeric(0))
  payoffs <- reactiveVal(list("left" = 0, "right" = 0))
  selected_arm <- reactiveVal("initial")
  flipped <- reactiveVal(numeric(0))
  game_stage <- reactiveVal("instructions")

  ## Game stage handlers ----

  # Update game stage based on trial number
  observeEvent(trial(), {
    if (trial() > num_trials) {
      game_stage("game_over")
    }
  })

  # Show final score
  output$final_score <- renderText({
    paste("Your total score:", sum(payouts()))
  })

  # Handle start button click
  observeEvent(input$start_button, {
    game_stage("in_progress")
    previous_pull_time(Sys.time())
  })

  ## Button handlers ----

  # Handle continue button click
  observeEvent(input$continue_button, {
    game_stage("results")
  })

  # Handle end button click
  observeEvent(input$end_button, {
    game_stage("end")
  })

  # Make game stage available to UI
  output$game_stage <- reactive({
    game_stage()
  })
  outputOptions(output, "game_stage", suspendWhenHidden = FALSE)

  ## UI renderers ----

  # Show trial number
  output$trial_num <- renderText({
    paste("Trial:", trial(), "/", num_trials)
  })

  # Show bandit image
  output$bandit_image <- renderImage(
    {
      list(
        src = bandit_image(selected_arm()),
        alt = "Two-Armed Bandit"
      )
    },
    deleteFile = FALSE
  )

  # Conditionally show result image
  output$outcome_animation <- renderUI({
    tags$img(id = "payoff_gif", src = "payoff.gif", alt = "Payoff", height = "100px", style = "visibility: hidden;")
  })

  # Display payoff
  output$payoff <- renderText({
    if (trial() > 1) {
      paste("+", payouts()[trial() - 1])
    } else {
      "Pull an arm to start!"
    }
  })

  # Display accumulated points
  output$total_points <- renderText({
    if (trial() > 1) {
      paste("Total score:", sum(payouts()[1:(trial() - 1)]))
    } else {
      "Total score: 0"
    }
  })

  ## Game logic ----

  # Update payoffs and flip status
  observeEvent(trial(), {
    if (trial() >= 1) {
      this_trial <- trial()
      if (this_trial == 1) {
        this_flipped <- sample(c(1, 2), 1)
      } else {
        prev_flipped <- flipped()[this_trial - 1]
        if (runif(1) < flip_prob) {
          this_flipped <- ifelse(prev_flipped == 1, 2, 1)
        } else {
          this_flipped <- prev_flipped
        }
      }
      base_payoff <- sample(payoff_range[1]:payoff_range[2], 1)
      payoff_values <- cbind(
        left = ifelse(this_flipped == 1, base_payoff + flip_bonus, base_payoff),
        right = ifelse(this_flipped == 2, base_payoff + flip_bonus, base_payoff)
      )
      payoffs(payoff_values)
      flipped(c(flipped(), this_flipped))
    }
  })

  # Handle left arm pull
  observeEvent(input$left_arm, {
    .arm_pull(1)
    session$sendCustomMessage(type = "resetGIF", message = list(src = "payoff.gif"))
  })

  # Handle right arm pull
  observeEvent(input$right_arm, {
    .arm_pull(2)
    session$sendCustomMessage(type = "resetGIF", message = list(src = "payoff.gif"))
  })

  ## Data logging ----

  # Track choices, payoffs, and flips for each trial
  game_data <- reactive({
    num_trials <- min(length(choices()), length(payouts()), length(flipped()))
    data.frame(
      trial = seq_len(num_trials),
      choice = choices()[1:num_trials],
      reaction_time = reaction_times()[1:num_trials],
      payout = payouts()[1:num_trials],
      flipped = flipped()[1:num_trials]
    )
  })

  ## Results renderers ----

  observeEvent(game_stage(), {
    if (game_stage() == "results") {
      # Get game data with choices and outcomes
      game_df <- game_data()

      ### Prepare results ----

      # Fit model to game data
      fit <- get_fit(game_df, m, lim)

      # Get estimated parameters
      params <- fit[-1]

      # Get model estimates
      value <- kalman_ucb(params, game_df, returns = "value")

      # Combine game data with model estimates
      result <- inner_join(game_df, value, by = c("trial", "choice"))

      # Create data for shaded areas
      shaded_df <- result %>%
        select(trial, choice) %>%
        distinct() %>%
        mutate(xmin = trial - 0.5, xmax = trial + 0.5)

      # Create dataframe for flip points
      flipped_df <- result %>%
        mutate(prev_flipped = lag(flipped, default = flipped[1])) %>%
        filter(flipped != prev_flipped) %>%
        select(trial)

      # Define arm colors and labels
      arm_colors <- c("#3498db", "#e74c3c") # blue for left (1), red for right (2)
      arm_labels <- c("1" = "Left Arm", "2" = "Right Arm")

      ### Arm selection ----

      # Calculate optimal arm percentage
      optimal_arm_percentage <- result %>%
        mutate(optimal_arm = ifelse(reward.1 > reward.2, 1, 2)) %>%
        summarize(percentage = round(100 * sum(choice == optimal_arm) / n(), 2)) %>%
        pull(percentage)

      # Create arm selection distribution plot
      arm_selection_plot <- game_df %>%
        count(choice) %>%
        mutate(
          choice = factor(choice, levels = c(1, 2)),
          freq = round(n / num_trials, 2)
        ) %>%
        ggplot(aes(x = choice, y = n, fill = choice)) +
        geom_col(width = 0.3, alpha = 0.7) +
        geom_text(aes(label = sprintf("%1.1f%%", freq * 100)), vjust = -0.25) +
        scale_x_discrete(labels = arm_labels) +
        scale_y_continuous(breaks = scales::pretty_breaks(), limits = c(0, nrow(game_df) + 2)) +
        scale_fill_manual(values = arm_colors, labels = arm_labels, breaks = c(1, 2)) +
        labs(
          title = "Arm Selection Distribution",
          subtitle = paste0("Optimal arm choice frequency: ", optimal_arm_percentage, "%"),
          x = "Arm", y = "Count"
        ) +
        theme_linedraw() +
        theme(
          aspect.ratio = 1,
          legend.position = "none",
          plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5)
        )

      ### Model estimates ----

      # Create plot for UCB values, and arm choice probabilities
      arm_ucb_plot <- result %>%
        left_join(shaded_df, by = c("trial", "choice")) %>%
        ggplot(aes(x = trial)) +
        geom_vline(data = flipped_df, aes(xintercept = trial), linetype = "dashed") +
        geom_rect(
          aes(
            xmin = xmin, xmax = xmax,
            ymin = -Inf, ymax = Inf,
            fill = factor(choice)
          ),
          color = "transparent", alpha = 0.5
        ) +
        geom_line(aes(y = ucb.1, color = as.factor(1)), linewidth = 1) +
        geom_line(aes(y = ucb.2, color = as.factor(2)), linewidth = 1) +
        scale_x_continuous(breaks = scales::pretty_breaks()) +
        scale_color_manual(values = arm_colors, labels = arm_labels) +
        scale_fill_manual(values = arm_colors, labels = arm_labels, breaks = names(arm_labels)) +
        labs(title = "UCB Value for Each Arm", x = "Trial", y = "UCB") +
        coord_cartesian(expand = FALSE) +
        theme_linedraw() +
        theme(aspect.ratio = 1, legend.position = "none", plot.title = element_text(hjust = 0.5))

      # Create plot for arm choice probabilities
      arm_probability_plot <- result %>%
        left_join(shaded_df, by = c("trial", "choice")) %>%
        ggplot(aes(x = trial)) +
        geom_vline(data = flipped_df, aes(xintercept = trial), linetype = "dashed") +
        geom_rect(
          aes(
            xmin = xmin, xmax = xmax,
            ymin = -Inf, ymax = Inf,
            fill = factor(choice)
          ),
          color = "transparent", alpha = 0.5
        ) +
        geom_line(aes(y = prob.1, color = as.factor(1)), linewidth = 1) +
        geom_line(aes(y = prob.2, color = as.factor(2)), linewidth = 1) +
        scale_x_continuous(breaks = scales::pretty_breaks()) +
        scale_color_manual(values = arm_colors, labels = arm_labels) +
        scale_fill_manual(values = arm_colors, labels = arm_labels, breaks = names(arm_labels)) +
        labs(title = "Probability of Choosing Each Arm", x = "Trial", y = "Probability") +
        coord_cartesian(expand = FALSE) +
        theme_linedraw() +
        theme(aspect.ratio = 1, legend.position = "none", plot.title = element_text(hjust = 0.5))

      ### Reaction time ----

      # Create reaction time plot
      reaction_time_plot <- game_df %>%
        left_join(shaded_df, by = c("trial", "choice")) %>%
        ggplot(aes(x = trial, y = reaction_time)) +
        geom_vline(data = flipped_df, aes(xintercept = trial), linetype = "dashed") +
        geom_rect(
          aes(
            xmin = xmin, xmax = xmax,
            ymin = -Inf, ymax = Inf,
            fill = factor(choice)
          ),
          color = "transparent", alpha = 0.5
        ) +
        geom_line(color = "#2ecc71", linewidth = 1) +
        scale_x_continuous(breaks = scales::pretty_breaks()) +
        scale_fill_manual(values = arm_colors, labels = arm_labels, breaks = names(arm_labels)) +
        labs(title = "Reaction Times", x = "Trial", y = "Reaction Time (s)") +
        coord_cartesian(expand = FALSE) +
        theme_linedraw() +
        theme(
          aspect.ratio = 1,
          legend.position = "none",
          plot.title = element_text(hjust = 0.5)
        )

      ### Cumulative reward ----

      # Create cumulative reward plot
      cumulative_reward_plot <- result %>%
        arrange(trial) %>%
        mutate(cumulative_reward = cumsum(payout)) %>%
        left_join(shaded_df, by = c("trial", "choice")) %>%
        ggplot(aes(x = trial, y = cumulative_reward)) +
        geom_vline(data = flipped_df, aes(xintercept = trial), linetype = "dashed") +
        geom_rect(
          aes(
            xmin = xmin, xmax = xmax,
            ymin = -Inf, ymax = Inf,
            fill = factor(choice)
          ),
          color = "transparent", alpha = 0.5
        ) +
        geom_line(color = "#f1c40f", linewidth = 1) +
        scale_x_continuous(breaks = scales::pretty_breaks()) +
        scale_fill_manual(values = arm_colors, labels = arm_labels, breaks = names(arm_labels)) +
        labs(title = "Cumulative Reward", x = "Trial", y = "Cumulative Reward") +
        coord_cartesian(expand = FALSE) +
        theme_linedraw() +
        theme(
          aspect.ratio = 1,
          legend.position = "none",
          plot.title = element_text(hjust = 0.5)
        )

      ### Regret ----

      # Create regret plot
      cumulative_regret_plot <- result %>%
        mutate(
          max_reward = pmax(reward.1, reward.2),
          regret = max_reward - payout,
          cumulative_regret = cumsum(regret)
        ) %>%
        left_join(shaded_df, by = c("trial", "choice")) %>%
        ggplot(aes(x = trial, y = cumulative_regret)) +
        geom_vline(data = flipped_df, aes(xintercept = trial), linetype = "dashed") +
        geom_rect(
          aes(
            xmin = xmin, xmax = xmax,
            ymin = -Inf, ymax = Inf,
            fill = factor(choice)
          ),
          color = "transparent", alpha = 0.5
        ) +
        geom_line(color = "#9b59b6", linewidth = 1) +
        scale_x_continuous(breaks = scales::pretty_breaks()) +
        scale_fill_manual(values = arm_colors, labels = arm_labels, breaks = names(arm_labels)) +
        labs(title = "Cumulative Regret", x = "Trial", y = "Cumulative Regret") +
        coord_cartesian(expand = FALSE) +
        theme_linedraw() +
        theme(
          aspect.ratio = 1,
          legend.position = "none",
          plot.title = element_text(hjust = 0.5)
        )

      # Create common legend
      common_legend <- guide_legend(
        title = "Arm",
        override.aes = list(color = arm_colors),
        labels = arm_labels
      )

      # Combine main plots into a figure with common legend
      fig <- (arm_selection_plot + arm_ucb_plot + arm_probability_plot +
        reaction_time_plot + cumulative_reward_plot + cumulative_regret_plot) +
        plot_layout(guides = "collect") &
        theme(legend.position = "bottom") &
        guides(color = common_legend, fill = "none")

      # Render combined plot
      output$results_fig <- renderPlot(
        {
          fig
        },
        width = 800,
        height = 600
      )

      ### Download handler ----

      output$download_data <- downloadHandler(
        filename = function() {
          paste0("game_data_", Sys.Date(), ".csv")
        },
        content = function(file) {
          write.csv(result, file, row.names = FALSE)
        }
      )
    }
  })
}

# Run app -----------------------------------------------------------------

shinyApp(ui, server)

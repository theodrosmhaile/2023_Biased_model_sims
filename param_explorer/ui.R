#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
shinyUI(
    fluidPage(

    # Application title
    titlePanel("Parameter Explorer"),
    
    h4("Select model and adjust sliders to visualize effect of parameter value on model outcome."),
    
    
    radioButtons(inputId = "modelselect", 
                       label= "Select Model to display", 
                       choices = c("RL only", "LTM only", "RL-LTM Strategy", "Meta-RL"), 
                       selected = "RL-LTM Strategy", inline = TRUE
                       
                       ),
    # Sidebar with a slider input for number of bins
    splitLayout( cellWidths = c("30%", "35%", "30%"),
        sidebarPanel( width = 10, fluid =TRUE,
            sliderInput("alpha",
                        "Learning Rate (Alpha):",
                        min = 0.05,
                        max = 0.25,
                        value = 0.05, 
                        step = 0.05, 
                        animate=TRUE),
     
            
            sliderInput("egs",
                        "RL noise (egs):",
                        min = 0.1,
                        max = 0.5,
                        value = 0.1, 
                        step = 0.1, 
                        animate = TRUE),
        
       
            sliderInput("mas",
                        "Attention/WM (MAS):",
                        min = 1.2,
                        max = 3.2,
                        value = 2, 
                        step = .2, 
                        animate = TRUE),
     
            sliderInput("ans",
                        "LTM Retrieval noise (ans):",
                        max = 0.5,
                        min = 0.1,
                        value = 0.1, 
                        step = 0.1, 
                        animate = TRUE),
        
            sliderInput("se",
                        "LTM decay rate (SE):",
                        min = 0.28,
                        max = 0.36,
                        value = 0.2, 
                        step = 0.02, 
                        animate = TRUE), 
            sliderInput("bias", 
                        "% RL Use:",
                        min = 20,
                        max = 80, 
                        value = 40, 
                        step = 20,
                        animate = TRUE
                        )
       ),
       
        # Show a plot of the generated distribution
       # mainPanel(
        plotOutput("curvPlot", width = "100%", height = "500px"), 
        plotOutput("barPlot", width = "70%", height = "500px")
         
            
        ),
  
   
    h4("The plotted curves and bars are averages of 100 simulations for each combination of parameters. "), 
     h4("The integrated model uses all the parameters."  ),
    
      h4("The RL model only uses the first two sliders: learning Rate and RL noise."), 
      h4("The LTM model takes input from the last three sliders: Attention/WM, LTM retrieval
      noise, and LTM decay rate."), 
         h4("Adjusting other input parameters will not produce an effect. "),
    br(),
    br()
    
   
    )

)#)

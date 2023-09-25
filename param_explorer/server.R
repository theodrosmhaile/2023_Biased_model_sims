#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#    http://shiny.rstudio.com/
#

library(shiny)
library(jsonlite)
library(tidyverse)

simsRL_LTM <- fromJSON('STR_sims_2023.JSON')$data %>%
    mutate(strtg=str_remove_all(strtg, "[:alpha:]") %>% 
               as.numeric())

simsRL     <- fromJSON('RL_sim_data_07_12_2022.JSON')$data
simsLTM    <- fromJSON('SE_LTM_sims_2023.JSON')$data

sims_meta  <- fromJSON('meta_rl_model_2023.JSON')$data
iter.n <- c(1:12)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
    
    output$curvPlot <- renderPlot({

# Based on chosen model, this selects the right data and set of parameters. 
        if (input$modelselect == "RL only") {
            temp <- simsRL
            ind  <- (temp$alpha    == input$alpha & 
                        # temp$se  == 0 &  
                         #temp$ans  == 0 &
                         temp$egs  == input$egs 
                        # temp$mas == 0
                         )
            
            
            set3.dat <- unlist(temp$set3_learn[ind])
            set6.dat <- unlist(temp$set6_learn[ind])
            
        }
        
        if (input$modelselect == "LTM only") {
            
            temp <- simsLTM
            ind  <- (temp$alpha    == 0 & 
                         temp$se  == input$se &  
                         temp$ans  == input$ans &
                         temp$egs  == 0 & 
                         temp$mas == input$mas)
            
            
            set3.dat <- unlist(temp$set3_learn[ind])
            set6.dat <- unlist(temp$set6_learn[ind])
        }
        
        if (input$modelselect == "RL-LTM Strategy") {
            
            temp <- simsRL_LTM
            
            ind  <- (temp$alpha    == input$alpha & 
                         temp$se  == input$se &  
                         temp$ans  == input$ans &
                         temp$egs  == input$egs & 
                         temp$mas == input$mas & 
                         temp$strtg == input$bias)
            
            
            set3.dat <- unlist(temp$set3_learn[ind])
            set6.dat <- unlist(temp$set6_learn[ind])
            
        }
        if (input$modelselect == "Meta-RL") {
            
            temp <- sims_meta
            ind  <- (temp$alpha    == input$alpha & 
                         temp$se  == input$se &  
                         temp$ans  == input$ans &
                         temp$egs  == input$egs & 
                         temp$mas == input$mas)
            
            
            set3.dat <- unlist(temp$set3_learn[ind])
            set6.dat <- unlist(temp$set6_learn[ind])
            
        }
         
        
        
        
# plot data points
        plot(iter.n, set3.dat, 
             col = '#e41a1c', 
             cex = 2, lwd = 2, 
             pch = 19, xlab ='Stimulus presentations',ylab ='Accuracy', ylim = c(0,1), 
             main = "Learning curves" )
        points(iter.n, 
             set6.dat, 
             col = '#377eb8', 
             cex = 2, lwd = 2, 
             pch = 19, xlab ='Stimulus presentations',ylab ='Accuracy', asp =c(3,4))
        
#draw fitted lines
        
        lines(iter.n, set3.dat, lwd=4.3, col = '#e41a1c')
        lines(iter.n, set6.dat, lwd=4.3, col = '#377eb8')
        
        legend("bottomright", c("set size 3", "set size 6"),pch = c( 19, 19),
               text.col =c( "#e41a1c","#377eb8"), col = c("#e41a1c","#377eb8"))
       
        
        
        output$barPlot <- renderPlot({
            
            
            barplot(c(temp$set3_test[ind], temp$set6_test[ind] ), 
                    c(1,1), ylab = 'Accuracy',
                    xlab = 'set size', 
                    col = c('#e41a1c', '#377eb8'),
                    ylim = c(0, 1), 
                    main = 'Test Accuracy')
        
        return(temp) 
        return(ind)
    })
    
   # output$barPlot <- renderPlot({
   #     
   # 
   #      barplot(c(temp$set3_test[ind], temp$set6_test[ind] ), 
   #              c(1,1), ylab = 'Accuracy',
   #              xlab = 'set size', 
   #              col = c('#e41a1c', '#377eb8'),
   #              ylim = c(0, 1), 
   #              main = 'Test Accuracy')

        
    })

})

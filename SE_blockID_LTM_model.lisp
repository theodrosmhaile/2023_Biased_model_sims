;;------------------------------------
;; LTM Model 2
;;------------------------------------
;;

;; This model relies only on storage and retrieval of memory of past experience
;; with stimuli and associated response.
;; It relies on three parameters: memory decay(BLL), activation noise(ANS) and
;; retrieval threshold(RT) at which a memory will be...activated/retrieved.
;; Important features: Stiumulus, associate-key and feedback

;;------------------------------------
;; Change log
;;------------------------------------
;; 03/30/20TMH updated commit-to-memory production to match integrated-model.lisp
;;            - Added parse-feedback-yes and parse-feedback-no productions
;;            - Enabled subsymbolic computations
;;            - Minor modifications to encode-feedback and commit-to-memory productions
;;            - Added parse-feedback-test productions
;; 
;; 08/18/2023 TMH this version:
;;            - deletes a chunk once in the retrieval buffer to prevent harvesting by Declarative module
;;            - has eblse enabled so that it uses the Pavlik spacing effect compuation. 
;;            - has a lower :mas parameter (formerly 8 and too high), now set to 2. 
;;            - while no changes are made here, this version is expected to work with an interface that uses a long break. 
;;            - while no changes are made here, this version is expected to work with an interface that presents stims at exact 2 sec intervals. 
;;
;; 08/28/2023 TMH this version:
;;            - utilizes an additional slot in stimulus that encodes Block ID. This is useful as the block id might be an additional source of 
;;              interference or fan effect. 
;;;           - added 'delete chunk' function in all response productions to prevent visual chunks from being written to DM to reduce fan effect. 
;;;
;;------------------------------------




(clear-all)

(define-model SE_LTM_model_blockID

(sgp ;;:bll 0.
     ;;:ans nil
     :er  t
     :ol nil
     :v nil
     :esc t
     ;:mas 2.0 ; fomerly 8
     :eblse t
     :act t
     ;:se-intercept 0.3
     :visual-activation 2
     )

;;---------------------------------------
;; Chunk types
;;---------------------------------------

(chunk-type goal
            responded ;; Whether a response was chosen or not
            fproc)    ;; fproc= feedback processed

(chunk-type stimulus
            picture
            associated-key
            outcome
            block_ID
            )

(chunk-type feedback
            feedback)

;;---------------------------------------
;; Chunks
;;---------------------------------------

;;(add-dm (yes isa chunk)
 ;;       (no  isa chunk)
   ;;     (make-response isa       goal
     ;;                  responded no
       ;;                fproc     yes)
        ;)


;;----------------------------------------
;; productions
;;----------------------------------------
   ;; Check memory: picture cur_pic, current picture presented is a variable.
   ;; This is a general purpose production that just takes in whatever presented stimulus
   ;; and checks against declarative memory in the retrieval buffer

(p check-memory
  
   =visual>
     picture =cur_pic
     block_ID =curr_block

   ?visual>
     state free

   ?imaginal>
     state free
     buffer empty

   =goal>
     fproc yes

   ?retrieval>
     state free
   - buffer full
  ==>

   +retrieval>
      picture =cur_pic
      outcome yes
     ; block_ID =curr_block

   +imaginal>
      picture =cur_pic
      block_ID =curr_block

   =visual>
)
;;-------------------------------------
;; Depending on outcome: yes or no (retrieval error)

   ;;outcome is no (retrieval error): make random response (3 possible)
;;-------------------------------------

(p response-monkey-j
  ?retrieval>
    state error

  ?imaginal>
    state free

  =imaginal>
    associated-key nil

  =goal>
    fproc yes

  =visual>
  - picture nil

  ?manual>
    preparation free
    processor free
    execution free
==>
!eval! (erase-buffer 'visual)
  +manual>
    cmd punch
    hand right
    finger index

  =imaginal>
    associated-key j

  *goal>
    fproc no

 ; =visual>
  )

(p response-monkey-k
  ?retrieval>
    state error

  =goal>
    fproc yes

  =visual>
  - picture nil

  ?imaginal>
    state free

  =imaginal>
    associated-key nil

  ?manual>
    preparation free
    processor free
    execution free
==>
!eval! (erase-buffer 'visual)
  +manual>
    cmd punch
    hand right
    finger middle

  =imaginal>
    associated-key k

  *goal>
    fproc no

 ; =visual>
  )


(p response-monkey-l
  ?retrieval>
    state error

  =visual>
  - picture nil

  ?imaginal>
    state free

  =goal>
    fproc yes

  =imaginal>
    associated-key nil

  ?manual>
    preparation free
    processor free
    execution free

==>
!eval! (erase-buffer 'visual)
  +manual>
    cmd punch
    hand right
    finger ring

  =imaginal>
    associated-key l

  *goal>
    fproc no

 ; =visual>
  )

;;-------------------------------------
;;outcome is yes: make response based on memory
;;-------------------------------------

(p outcome-yes
  =retrieval>
    outcome yes
    associated-key =k

  =goal>
    fproc yes

  ?imaginal>
    state free

  =imaginal>
    associated-key nil

  ?manual>
    preparation free
    processor free
    execution free

==>

!eval! (erase-buffer 'retrieval )
!eval! (erase-buffer 'visual)
  +manual>
    cmd press-key
    key =k

  *imaginal>
    associated-key =k

  *goal>
    fproc no
)


;;Encode response after feedback

(p parse-feedback-yes
   =visual>
     feedback yes

   ?visual>
     state free

   =goal>
    fproc no
==>
   =visual>

   *goal>
     fproc yes
)


(p parse-feedback-no
   =visual>
     feedback no

   ?visual>
     state free

   =goal>
      fproc no
==>
   =visual>

   *goal>
     fproc yes
)


(p encode-feedback
   "Encodes the visual response"
  =visual>
    feedback =f

  ?imaginal>
    state free

  =goal>
    fproc yes

  =imaginal>
    outcome nil

==>
  *imaginal>
    outcome =f

  ;*goal>
   ; fproc yes

  =visual>
  )


(p commit-to-memory
   "Creates an episodic traces of the previous decision"
  =visual>
    feedback =f

  =goal>
    fproc  yes

  =imaginal>

  - outcome nil

==>



  -visual>
  -imaginal>
)

(p parse-test-feedback
   =visual>
     feedback x

   ?visual>
     state free

   =goal>
       fproc no
==>
   =visual>

   *goal>
     fproc yes
)

;;(p outcome-no-commit-to-memory

  ;;)

;;(goal-focus make-response)



;;(set-buffer-chunk 'visual 'cup-stimulus)

)

;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      : Dan Bothell
;;; Copyright   : (c) 2016 Dan Bothell
;;; Availability: Covered by the GNU LGPL, see LGPL.txt
;;; Address     : Department of Psychology 
;;;             : Carnegie Mellon University
;;;             : Pittsburgh, PA 15213-3890
;;;             : db30@andrew.cmu.edu
;;; 
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    : dispatcher.lisp
;;; Version     : 3.1
;;; 
;;; Description : 
;;; 
;;; Bugs        : 
;;;
;;; To do       : [ ] Kill threads that aren't needed instead of just letting them
;;;                   run to completion.  
;;; 
;;; ----- History -----
;;; 2016.11.16 Dan [1.0]
;;;             : * Actually put the header on the file and for now have the
;;;             :   initialization code to test for quicklisp and just error
;;;             :   out if it's not available.
;;;             : * Changed the add command so that if single-instance is a string
;;;             :   then instead of waiting for the lock if it's running it just
;;;             :   fails and sends that string as the message.
;;;             : * Added the commands for output and trace monitoring.
;;; 2016.11.17 Dan
;;;             : * Be more careful about the return values from the Lisp side
;;;             :   interface functions because when there's an error the params
;;;             :   will just be a string.
;;;             : * Changed execute-act-r-command to evaluate-act-r-command so
;;;             :   that it's more consistent with the external interface and
;;;             :   changed stop-monitoring to remove-act-r-command-monitor
;;; 2016.11.18 Dan
;;;             : * Moved the load-act-r-model command here from the load file
;;;             :   and now capture all the output to send to the warning trace
;;;             :   or error out if the file doesn't exist.
;;; 2016.11.22 Dan
;;;             : * To pass symbols back and forth for now going to use an object
;;;             :   in the JSON arrays of the form {"name":"<symbol-name>"}.
;;;             :   Adding modifications to the encoding and decoding to handle
;;;             :   that automatically.
;;;             : * Fixed an issue with parsing of the error messages out of a
;;;             :   reply.
;;; 2016.11.23 Dan
;;;             : * Added a format function for printing the error messages so
;;;             :   that ACL and CCL will show all the details (may need to add
;;;             :   additional Lisps as well as needed).
;;;             : * Added the cleaner encode-json symbol method which only does
;;;             :   the symbol->object when a variable is set.
;;; 2016.12.01 Dan
;;;             : * Changed load-act-r-model command to error if already evaluating.
;;;             : * Don't attempt to do any fancy encoding/decoding for symbols
;;;             :   now -- let it use the default camel case stuff for decoding
;;;             :   because at this point the only thing converted to symbols 
;;;             :   will be the object slot names which have fixed values. 
;;;             :   However, the lisp symbol to string needs to not use the
;;;             :   camel case converter otherwise something like visual-location
;;;             :   gets converted to "visualLocation" which is bad.
;;;             :   This does potentially cause problems if someone tries to pass
;;;             :   an object "through" the system (parameters to a function that
;;;             :   was added by something else) but for now not worrying about
;;;             :   that.
;;; 2016.12.02 Dan
;;;             : * Changed how the print-error-message function handles things
;;;             :   for ACL and CCL to just write the error message because 
;;;             :   that seems to work better across different error/condition
;;;             :   types.
;;; 2016.12.09 Dan
;;;             : * Adding a new method to the dispatcher - check.  It takes
;;;             :   one parameter which is a string.  If that string names a
;;;             :   command that exists then it returns three values: t,
;;;             :   a bool indicating whether the caller is the owner, and the
;;;             :   documentation string.  If it is not a command it returns the
;;;             :   single value nil (which probably ends up as none in JSON).
;;; 2016.12.14 Dan
;;;             : * Added the decode-string-names function to parse strings
;;;             :   into symbols where a string that starts with a : will be
;;;             :   upcased and interned into the keyword package without the
;;;             :   initial colon, a string which starts and ends with a ' will
;;;             :   be converted into a string without the first and last chars,
;;;             :   all other strings will be upcased and interned in the package
;;;             :   stored in *default-package*.  Other items which are valid 
;;;             :   values are t, nil, numbers, and symbols which will be left 
;;;             :   unchanged.  If a list (or nested lists) of items is provdided
;;;             :   it will be recursively processed to create a list of items
;;;             :   as above.  If an invalid item is provided or an unprocessable
;;;             :   item is encountered (a single quote for example since the
;;;             :   first and last characters are ') then an error will be 
;;;             :   generated.
;;; 2016.12.19 Dan
;;;             : * The feature test for ACL is allegro not acl -- fixed that
;;;             :   for the print-error-message function.
;;; 2017.01.18 Dan
;;;             : * Change echo-act-r-output so that it just sets the monitors
;;;             :   and directs the output to the current *standard-output*.
;;;             :   To go along with that are suppress-act-r-output and turn-off-
;;;             :   act-r-output.  The first just directs output to nil as an
;;;             :   "around" macro whereas the second is a function that removes
;;;             :   the monitors and requires echoing again.
;;;             : * Added handle-evaluate-results to deal with printing errors
;;;             :   and returning the results for the direct function calls.
;;; 2017.01.30 Dan
;;;             : * Changed add-act-r-command parameter order because it should
;;;             :   match the client and unfortunately the client version is in
;;;             :   a "better" order for users...
;;;             : * Updated the process-external-input function to better handle
;;;             :   the distinction between an invalid command method and an 
;;;             :   error in the parameters provided.
;;; 2017.02.08 Dan
;;;             : * Return-result better handles when it can't encode the return
;;;             :   result in JSON and still returns an error result indicating
;;;             :   that's what happened.
;;;             : * Change the default command monitor to a new style that I'm
;;;             :   calling 'simple'.  It is called after the afters and only
;;;             :   passed the parameters sent to the monitored command.
;;;             : * The trace monitor is now just a simple function.
;;; 2017.02.15 Dan
;;;             : * Added the dispatch-apply function.
;;;             : * Added the encode-string function to go along with decode-string.
;;; 2017.02.16 Dan
;;;             : * Fixed a bug in dispatch-apply.
;;; 2017.02.17 Dan
;;;             : * Added local-or-remote-function-p.
;;;             : * Added local-or-remote-function-or-nil for use in hook tests.
;;; 2017.02.22 Dan
;;;             : * Added dispatch-apply-list because sometimes the items are
;;;             :   already in a list which the &rest of dispatch-apply would
;;;             :   wrap in another list.
;;; 2017.02.27 Dan
;;;             : * Need to add the process-events function since that was only
;;;             :   defined on the client side.
;;; 2017.03.13 Dan
;;;             : * Fixed a bug with turn-off-act-r-output.
;;; 2017.03.17 Dan [2.0]
;;;             : * Creating threads is a significant portion of the cost for
;;;             :   the dispatching (it looks like ~40% in some cases) so now
;;;             :   I'm going to change it to have a pool of 'worker' threads
;;;             :   that can run things.  If there isn't an available worker
;;;             :   then it will create a new one.  To keep things simple with
;;;             :   respect to detecting whether there's an available worker
;;;             :   I'm just going to keep a list of them around and pass it
;;;             :   off explicitly to the next available one instead of using
;;;             :   a condition variable to let 'anyone' grab it.  Then it is
;;;             :   up to the worker to put itself back into the available list
;;;             :   upon completion.  There are actually two sets of workers 
;;;             :   needed -- those for handling the 'action' in the thread that's
;;;             :   running process-external-input for a connection and those for
;;;             :   evaluating a command as needed in the perform-action function
;;;             :   called from the dispatcher-process-actions thread.
;;; 2017.03.20 Dan
;;;             : * Working version of the first set of worker threads implemented.
;;;             : * Added the second set of worker threads as well.
;;;             : * Seems to have a noticeable improvement for most tasks tested.
;;; 2017.03.28 Dan
;;;             : * Moved the quicklisp loading of the libraries to the main
;;;             :   load file.
;;; 2017.03.31 Dan
;;;             : * Added the stop-des function and save the main connection
;;;             :   socket in the dispatcher to let that happen cleanly.
;;; 2017.04.05 Dan
;;;             : * In order to support multiple models and better parallel
;;;             :   operation of commands the evaluate-act-r-command function
;;;             :   now requires that *current-act-r-model* will be set and it
;;;             :   records that struct to rebind to *current-act-r-model* in the
;;;             :   thread which actually performs the evaluation.
;;;             : * The remote connections now need to provide at least two
;;;             :   parameters for evaluate: a command and a model name.  If that
;;;             :   is an internal command then it will bind *current-act-r-model*
;;;             :   to the struct for the named model when it evaluates the command.
;;;             :   Otherwise it will just pass it along to the appropriate owner.
;;;             : * That passing along of commands now requires clients only
;;;             :   supply 1 method -- evaluate.  That method needs to work like
;;;             :   the dispatcher instead of having the "interface" handle things.
;;;             :   That way the recording of "current model" can be done in
;;;             :   whatever way seems appropriate on the other side.
;;;             : * Added the get-model function to return a model structure
;;;             :   given a string or symbol name for the model.
;;; 2017.06.02 Dan
;;;             : * Handle-evaluate-results now uses print-warning instead of
;;;             :   printing to *error-output*.
;;;             : * Change how *current-act-r-model* is used -- don't use that
;;;             :   directly in evaluate.  Use (current-model-struct) to set the
;;;             :   current model instead.
;;; 2017.06.15 Dan
;;;             : * Make print-warning thread safe.
;;; 2017.06.22 Dan
;;;             : * Make get-model safe by protecting meta-p-models.
;;; 2017.06.28 Dan 
;;;             : * Have load-act-r-model go through the dispatcher.
;;; 2017.07.13 Dan
;;;             : * Add the general-trace and act-r-output to support output 
;;;             :   that can happen without a model defined, and update the echo
;;;             :   and suppress functions to deal with it too.
;;;             : * Moved the print-error-message function to misc-utils since
;;;             :   that's a better place for it for general use.
;;; 2017.07.14 Dan
;;;             : * Cleaned up some unneeded declarations.
;;;             : * Added the with-top-level-lock macro to provide a way to keep
;;;             :   high-level commands from running simultaneously (reset, reload,
;;;             :   run, etc).
;;; 2017.07.20 Dan
;;;             : * Give all the threads names for debugging purposes.
;;;             : * Name all the locks and cvs too!
;;;             : * Start to fix a race condition with thread startup that doesn't
;;;             :   happen on my main development machine...
;;;             : * Got all the startup thread issues squared away, and also fixing
;;;             :   some notifys that aren't happening with the lock held.
;;; 2017.08.22 Dan
;;;             : * Changed the error message for invalid methods to include the
;;;             :   check method too.
;;; 2017.08.23 Dan
;;;             : * Added the remote command for getting the version string.
;;;             : * Provide a better error message when an invalid form of a
;;;             :   method is given.
;;;             : * Allow an added command to have no function and when evaluated
;;;             :   it doesn't call anything.  That is purely for monitoring
;;;             :   purposes and simplifies things like model-trace needing a 
;;;             :   dummy function and should improve performance.
;;; 2017.08.25 Dan
;;;             : * Load-act-r-command now returns t if the load succeeded.
;;;             : * It also now has an optional for compiling the file too.
;;; 2017.08.29 Dan
;;;             : * Return-result didn't distinguish between a json parsing
;;;             :   error and a socket write error which could lead to problems
;;;             :   since the socket write got tried again which left the
;;;             :   worker stuck.
;;;             : * Check-act-r-command needs to return the result not the success
;;;             :   status.
;;;             : * Same for list-act-r-commands.
;;; 2017.09.07 Dan
;;;             : * Need to adjust the json encoder to make sure that keywords
;;;             :   are converted to strings that retain the colon.
;;; 2017.09.13 Dan
;;;             : * Added a separate load-act-r-code command since you can't
;;;             :   use load-act-r-model to load a file that calls load-act-r-
;;;             :   model, like the task files for the current tutorial, and
;;;             :   the Environment needs a way to load those task files.
;;; 2017.10.06 Dan
;;;             : * Added the *dont-start-dispatcher* variable which gets set
;;;             :   for the standalone versions and tested in the eval-when
;;;             :   because don't want to start it when building the apps.
;;;             : * Added dont-start-des and init-des to go along with that.
;;; 2017.10.13 Dan
;;;             : * Wrapped another handler-case around the input stream processing
;;;             :   read-char because in CCL under windows closing the stream
;;;             :   doesn't seem to be an EOF that is ignored.
;;;             : * Anything that writes to *error-output* now uses send-error-output
;;;             :   which includes a finish-output since CCL buffers that stream
;;;             :   (at least in the Windows version) for some insane reason!
;;; 2017.10.23 Dan
;;;             : * Actually set *act-r-echo-command* to t in echo-act-r-output
;;;             :   so that turn-off-act-r-output will actually stop things...
;;; 2017.11.01 Dan
;;;             : * Use the :standalone switch to determine whether or not to
;;;             :   start the dispatcher.
;;; 2017.12.05 Dan
;;;             : * Change how the return value is generated for the signals
;;;             :   because (cons t t) doesn't seem to send things out right,
;;;             :   but (list t t) does.
;;; 2017.12.12 Dan
;;;             : * Adding the option of creating a Lisp side macro for remotely
;;;             :   added commands using a new 5th parameter for the name.
;;;             :   Right now this seems to have a bad interaction with the ACL IDE
;;;             :   because it errors if you try to use the created macros in the
;;;             :   Debug window saying that the macros created aren't macros or
;;;             :   functions, but if you use them from the console window or
;;;             :   in code that's loaded they work fine.
;;; 2017.12.14 Dan
;;;             : * Added some saftey checks to the dispatch-apply-* functions
;;;             :   to avoid errors for non-function values.
;;;             : * Have the local-or-... tester return t or nil instead of a
;;;             :   generalized boolean result from the or.
;;; 2017.12.19 Dan
;;;             : * Fixed the dispatch-* functions so that they return multiple
;;;             :   values when appropriate.
;;;             : * Added a version of each dispatch-* called dispatch-*-names
;;;             :   which applies decode-string to the results before returning
;;;             :   them.
;;;             : * Added a Lisp function for load-act-r-code.
;;; 2017.12.19 Dan
;;;             : * Removed some of the condition clauses from the handler-case
;;;             :   calls because they were tripping on trivial issues when loading
;;;             :   a file remotely (like from the Environment).
;;; 2017.12.20 Dan
;;;             : * Fixed a bug with dispatch-eval because it could be passed a
;;;             :   non-list value.
;;; 2018.01.02 Dan
;;;             : * Fixed a bug that could lead to repeated output in Lisp from
;;;             :   echo-act-r-output and turn-off-act-r-output.
;;; 2018.01.24 Dan
;;;             : * Echo the output by default now if it's not the standalone.
;;; 2018.03.14 Dan
;;;             : * Added a command for getting a new command name.  It keeps
;;;             :   track of those it's provided so it can guarantee that they're
;;;             :   not used and clears them when added.  That command is called
;;;             :   "get-new-command-name" or get-new-command-name for Lisp.
;;; 2018.03.19 Dan
;;;             : * A ~ in the json result to send caused an error with format,
;;;             :   but fixed that now.
;;; 2018.04.05 Dan [2.1]
;;;             : * For the standalone, if it fails to open the default port it
;;;             :   will try incrementing until it can.  Once it does it will 
;;;             :   print the port used and write that number to a file in the
;;;             :   current home directory: ~/act-r-port-num.txt.
;;;             : * Loading from sources will also increment port until success
;;;             :   and also writes the port num file.
;;; 2018.04.06 Dan
;;;             : * Switch to using the "real" ip address so that external connections
;;;             :   can be allowed (but default to not by *allow-external-connections*).
;;;             : * Also writes the address to a file now too so that cleints can
;;;             :   connect appropriately (since one could still use 127.0.0.1
;;;             :   explicitly).
;;; 2018.04.24 Dan
;;;             : * Added a flag and access function to test whether a load is
;;;             :   coming through the dispatcher since error handlers can't
;;;             :   invoke the debugger from a background thread.
;;;             : * Turns out some systems might not get the "real" address and
;;;             :   end up with something like 127.0.1.1 which won't match the
;;;             :   incoming so also test for 127.0.0.1 and localhost directly
;;;             :   when checking for local connection.
;;; 2018.06.07 Dan
;;;             : * Added encode-string-names which is the recursive version of
;;;             :   encode-string. 
;;; 2018.06.11 Dan
;;;             : * Have decode-string-names use decode-string for consistency.
;;; 2018.06.12 Dan
;;;             : * Added a process-options-list function for internal use.
;;; 2018.06.13 Dan
;;;             : * Updated the command doc strings to match current spec.
;;; 2018.06.15 Dan
;;;             : * Added convert-options-list-items to post-process a valid
;;;             :   options list result and convert ind cated values with either
;;;             :   string->name-recursive or decode-string-names.
;;; 2018.06.28 Dan
;;;             : * Fixed a bug with convert-options-list-items because it needs
;;;             :   to return nil not a list of nil when there are no options.
;;; 2018.07.11 Dan
;;;             : * In addition to writing out the port and address in the home
;;;             :   directory write out a config file for the environment 
;;;             :   explicitly to override the defaults for when it can't read
;;;             :   the files (not sure why that happens, but seems to be an
;;;             :   issue on one of the Summer School student's machines).
;;; 2018.08.29 Dan
;;;             : * Stop-des needs to kill all of the worker threads too, and only
;;;             :   close the socket if it's actually open.
;;; 2018.08.30 Dan
;;;             : * usocket-p isn't external in usocket so need to use usocket::.
;;;             : * Also need to check all the threads before trying to kill them.
;;; 2018.11.09 Dan [3.0]
;;;             : * Fix a typo in the error for a check action.
;;;             : * Adding two new operations:
;;;             :    - list-connections : return details of all connections
;;;             :    - set-name : provide a name for the connection.
;;; 2018.11.14 Dan
;;;             : * More work towards the new operations.
;;; 2018.11.15 Dan
;;;             : * Finished off the two new commands.
;;; 2018.11.16 Dan [3.1]
;;;             : * Update the docs since set-name can be used repeatedly.
;;;             : * Internal status info returns the connection address, and
;;;             :   store the port in a global like the address.
;;; 2019.01.08 Dan
;;;             : * Cleaned up the IP address initialization because it had a
;;;             :   bad assumption.
;;;             : * Use the usocket:ip= instead of equalp when testing for the
;;;             :   connection comming from same machine.
;;; 2019.02.05 Dan
;;;             : * Adjustments because meta-p-models is now an alist.
;;; 2019.02.14 Dan
;;;             : * When defaulting to 127.0.0.1 set *allow-external-connections*
;;;             :   to true since locals may show the real address and externals
;;;             :   can't access the loopback address.
;;;             : * Before defaulting to 127.0.0.1 try looking up "localhost"
;;;             :   since it should be available and might not be 127.0.0.1.
;;; 2019.02.15 Dan
;;;             : * Added a switch to force it to use the local address, and
;;;             :   going to set that in the standalones because public wifi
;;;             :   networks often block machine-to-machine which can also 
;;;             :   block machine to itself.  So, have the standalones default
;;;             :   to that since external access is likely going to be rarely
;;;             :   used.
;;; 2019.03.22 Dan
;;;             : * Reorder dispatch-apply-list since it's called frequently
;;;             :   with function names...
;;; 2019.03.26 Dan
;;;             : * Don't create names for the locks in send-des-action.  Has a
;;;             :   huge perform hit (3% faster without in zbrodoff test).
;;; 2019.03.27 Dan
;;;             : * Removing all the names for dynamically created locks and
;;;             :   condition variables.
;;; 2019.04.09 Dan
;;;             : * Added support for the :single-threaded-act-r flag which
;;;             :   results in the dispatcher not starting any threads (thus no 
;;;             :   external connections possible) and all the locks being 
;;;             :   ignored.  
;;; 2019.04.10 Dan
;;;             : * Actually put the flags on start-des, init-des, and stop-des
;;;             :   as well to prevent anyone trying to start it in the single-
;;;             :   threaded build.
;;; 2019.05.28 Dan
;;;             : * Switch to using jsown to decode incoming messages and reworked
;;;             :   the message parser a little.
;;; 2019.05.30 Dan
;;;             : * Switch back to cl-json from jsown since jsown doesn't keep
;;;             :   the same number types as provided e.g. 1.0 --> 1 which is
;;;             :   an issue for things like act-r-random.
;;; 2019.06.17 Dan
;;;             : * Add a hack so that CCL's GUI sends all the output to the
;;;             :   initial listener window (or current if stopped and restarted)
;;;             :   instead of an AltConsole.
;;; 2019.06.20 Dan
;;;             : * Added another handler-case to catch any potential problems
;;;             :   when sending a command because otherwise that could result 
;;;             :   in a deadlock when that client exits.
;;; 2019.06.28 Dan
;;;             : * Printing the action-parameters can result in an infinite
;;;             :   loop if *print-circle* is nil so make sure it's t whenever
;;;             :   they're printed.
;;;             : * Change the create-local-macro command again, and have it
;;;             :   handle conditions (since various things can occur there)
;;;             :   so that 'safe' issues don't abort the command creation.
;;;             : * Bug in the processing of a returned error message.
;;; 2019.07.01 Dan
;;;             : * Added a fmakunbound to create-local-macro to prevent it from
;;;             :   throwing a condition and exiting to a higher level handler-
;;;             :   case.
;;;             : * Which doesn't stop LispWorks from wanting to still assert
;;;             :   a redefinition warning...  So, basically following the 
;;;             :   example from the muffile-warning docs to create a function
;;;             :   to invoke a restart to avoid the warning.
;;; 2019.09.09 Dan
;;;             : * Adjusted with-top-level lock to take another parameter that
;;;             :   indicates what is holding the lock so that the warning
;;;             :   message is a little more informative.
;;; 2019.09.10 Dan
;;;             : * Changed the perform-action for the single-threaded version
;;;             :   to not catch conditions for evaluate to avoid a problem with
;;;             :   loading test models.  Not sure why that doesn't trip up the
;;;             :   multi-threaded version, but presumably it's in how the 
;;;             :   handler cases get unwrapped between threads.
;;;             : * Added a start-des for the single threaded build which just
;;;             :   reports that it can't start one to avoid a warning.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; General Docs:
;;; 
;;; This code provides a new central command dispatch system which has a TCP/IP
;;; server listening on 127.0.0.1:2650 and will accept connections for remote
;;; access to that command dispatch system.  There are 7 operations available
;;; through the command dispatch system:
;;;
;;; evaluate: Given a command and a list of parameters sends those to the owner
;;;           of that command for evaluation and returns the list of results from
;;;           that evaluation.
;;; add: Specify the name of a command to be added to those available through the
;;;      central dispatch system.  Also requires specifying the name which gets
;;;      sent to the command owner for handling it (it doesn't have to match the
;;;      command name provided to other systems).  Optionally, can provide a 
;;;      string which documents the command, a flag to indicate that only one
;;;      instance of the command should be evaluated at a time, and a name for
;;;      a Lisp side macro to call the command.
;;; remove: Specify a command to remove from the central dispatch system.
;;; monitor: Specify a command which will monitor another command.  When the
;;;          monitored command is evaluated the monitoring command will also be
;;;          evaluated.  The monitoring command can be specified to be evaluated
;;;          either before or after the monitored command.  If it is evaluated
;;;          before then it will be called with 4 parameters which are the name
;;;          of the monitored command, the list of parameters which will be sent
;;;          to the monitored command, and then two empty lists.  If it is after
;;;          the monitored command then the first two parameters are the same as
;;;          in the before case, the third indicates whether the monitored command
;;;          completed successfully, and the fourth is the list of values that
;;;          the monitored command returned if successful or the error message
;;;          it returned if not successful.
;;;          A new addition to monitoring is a simple mode.  In simple mode
;;;          the monitor is called after, but is just passed the parameter list
;;;          that the original command received.  That is now the default if neither
;;;          :before or :after isn't specified.
;;; remove-monitor: Stop the monitoring of a command set up with the monitor 
;;;                 operation.  Requires the monitored command and the monitoring
;;;                 command to remove from it.
;;; list-commands: Takes no parameters and returns a list of lists.  Each of the
;;;                sublists contains two elements: the name of a command and its
;;;                documentation string.
;;; check: Takes one parameter and returns one or three values.  If the parameter
;;;        names a command which has been added then return three values: t,
;;;        a bool indicating whether the caller is the owner (or nil if it
;;;        is a reserved name), and the documentation string (or "Reserved"
;;;        if it is a reserved name).  If it is not a command it returns the
;;;        single value nil.
;;; set-name: Takes one parameter and returns one value.  If the parameter is
;;;        a string then that string is recorded as the current name of the 
;;;        connection (does not have to be unique) and t is returned.  Otherwise 
;;;        it returns nil.
;;; list-connections: Takes no parameters.  Returns a list of lists where each
;;;        sublist represents a connection to the dispatcher and contains 5 items:
;;;        connection name (or nil), ip-address string, count of pending calls,
;;;        count of pending evaluations, and list of commands.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Public API:
;;;
;;; JSON-RPC 1.0 communication over a socket stream using char-code 4 to indicate
;;; end of message for remote connections < fill in more details >.
;;;
;;; For Lisp side there are functions for accessing 8 of the available operations:
;;;  add-act-r-command (name function &optional documentation single-instance local-name)
;;;  remove-act-r-command (name)
;;;  evaluate-act-r-command (name &rest rest)
;;;  monitor-act-r-command (act-r-command monitor &optional when)
;;;  remove-act-r-command-monitor (act-r-command monitor)
;;;  list-act-r-commands ()
;;;  check-act-r-command(name)
;;;  list-act-r-connections()
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Design Choices:
;;; 
;;; The main idea is that everything must pass through the central command dispatch
;;; system so that from a high-level view it doesn't matter how/where a particular
;;; component is implemented.  At this point, that's not fully realized because
;;; a lot of the Lisp code that handles things like modules and multiple models is
;;; not available through the central dispatcher and I haven't really come up with
;;; a way to do something like (with-model foo ...) through the central system yet.
;;;
;;; Use systems available through QuickLisp to handle the threading, sockets, and
;;; JSON parsing.  For now, load those using quickload, but consider rolling them
;;; in at some point so it can be loaded without QuickLisp.
;;;
;;; Chose to use JSON-RPC 1.0 (at least a subset of it) for the communication
;;; interface because it specifies a peer-to-peer connection, whereas the newer
;;; JSON-RPC 2.0 is a client-server system which would basically require each
;;; remote system to provide both roles to really do anything which seems to
;;; be more complicated than it needs to be.  However, could change that if I
;;; find that existing libraries for other languages handle that cleanly.
;;; 
;;; All errors/problems write to *error-output*, but may want some other sort of
;;; error logging eventually.  Everything should be wrapped with enough handler-case
;;; and unwind-protect code to keep it from crashing while handling the external
;;; and internal communications, but nothing is ever 100% safe...
;;;
;;; There are potentially a lot of threads generated for this:
;;;  - one thread for the listening socket which spawns new threads for each
;;;    connection that is made
;;;  - a thread monitoring the action queue which spawns a new thread for each
;;;    action that needs to be processed
;;;
;;;  This code is peppered with enough locks to keep things thread safe, but 
;;;  the command code which gets evaluated needs to protect itself as well.  Just
;;;  specifying a single instance might be enough, but it should be careful of
;;;  dependencies if it calls other commands too.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; The code
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#+:packaged-actr (in-package :act-r)
#+(and :clean-actr (not :packaged-actr) :ALLEGRO-IDE) (in-package :cg-user)
#-(or (not :clean-actr) :packaged-actr :ALLEGRO-IDE) (in-package :cl-user)

;;; Record the package for use in the threads that are created

(defparameter *default-package* *package*)

;;; Need to get models based on name

(defun get-model (name)
    "Returns the model struct if name is the name of a model in the current meta-process - there must be a current mp"
  (let ((mp (current-mp)))
    (bt:with-lock-held ((meta-p-models-lock mp))
      (or (cdr (assoc name (meta-p-models mp)))
          (and (stringp name) (cdr (assoc (string->name name) (meta-p-models mp))))))))

;;; The main control 

(defstruct dispatcher
  connection-interface
  connection-socket
  action-thread
  action-queue
  (command-table (make-hash-table :test 'equal))
  (spec-names (make-hash-table :test 'equal))
  connections
  action-lock
  action-cv
  handler-lock
  command-lock
  available-received-lock
  available-received-workers
  available-execute-lock
  available-execute-workers)

(defstruct worker lock cv thread handler action other started start-lock)

(defvar *dispatcher* nil)
#+:standalone (defvar *dont-start-dispatcher* t)
#-:standalone (defvar *dont-start-dispatcher* nil)

(defstruct action
  type
  model
  parameters
  cv
  lock
  complete
  result-success
  result
  evaluator)

(defvar *dispatch-command* 0)

(defstruct dispatch-command
  name documentation 
  underlying-function 
  evaluator 
  local-name
  
  ;; these are lists of dispacth-command structures
  before 
  after 
  simple
  monitoring
  ;;;
  
  single-instance
  (lock (bt:make-lock )))

(defvar *pr-lock* 0)
(defvar *pr-cv* 0)

(defstruct pending-request id action (lock (bt:make-lock )) 
  (cv (bt:make-condition-variable ))
  complete success result)

(defvar *handler-lock* 0)
(defvar *handler-cv* 0)

(defstruct handler socket thread (stream-lock (bt:make-lock ))
  (sent-requests-lock (bt:make-lock ))
  sent-requests (received-requests-lock (bt:make-lock ))
  received-requests (command-lock (bt:make-lock )) commands (id 0)
  name)

(defvar *allow-external-connections* nil)

(defmacro send-error-output (control-string &rest args)
  `(progn
     (format *error-output* ,control-string ,@args)
     (finish-output *error-output*)))



(defun decode-string (s)
  (if (stringp s)
      (let ((l (length s)))
        (if (zerop l)
            (error "Empty string provided where value was required")
          (if (and (char= #\' (char s 0)) (> l 1) (char= #\' (char s (1- (length s)))))
              (subseq s 1 (1- (length s)))
            (string->name s))))
    s))

(defun encode-string (s)
  (if (stringp s)
      (format nil "'~a'" s)
    s))


;; Need to make sure keyword symbols keep the colon on the front
;; so can't just use the symbol-name of a symbol.

(defmethod json::encode-json ((s symbol) &optional (stream json::*json-output*))
  "Write the JSON representation of the symbol S to STREAM (or to
*JSON-OUTPUT*).  If S is boolean, a boolean literal is written.
Otherwise, the name of S is passed to *LISP-IDENTIFIER-NAME-TO-JSON*
and the result is written as String."
  (let ((mapped (car (rassoc s json::+json-lisp-symbol-tokens+))))
    (if mapped
        (progn (write-string mapped stream) nil)
        (let ((s (funcall json::*lisp-identifier-name-to-json* (if (keywordp s) (format nil "~s" s) (symbol-name s)))))
          (json::write-json-string s stream)))))


(setf json:*lisp-identifier-name-to-json* 'identity)


;; These will only be called from Lisp.
;; Externals will generate their action structs in the handler.


(defun add-act-r-command (name &optional function documentation (single-instance t)(local-name nil))
  (let ((a (make-action :type 'add-command 
                        :parameters (list :name name :underlying-function function
                                          :documentation documentation 
                                          :evaluator :lisp
                                          :single-instance single-instance
                                          :local-name local-name))))
   
    (multiple-value-bind (success params)
        (send-des-action a)
      (if (listp params)
          (values-list (cons success params))
        (values-list (list success params))))))


(defun remove-act-r-command (name)
  (let ((a (make-action :type 'remove-command 
                        :parameters (list name)
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if (listp params)
          (values-list (cons success params))
        (values-list (list success params))))))
  
(defun evaluate-act-r-command (name &rest rest)
  (let ((a (make-action :type 'execute-command
                        :model (current-model-struct)
                        :parameters (list :name name :parameters rest)
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if (listp params)
          (values-list (cons success params))
        (values-list (list success params))))))

(defmacro handle-evaluate-results (&body body)
  `(let ((result (multiple-value-list ,@body)))
     (if (first result)
         (values-list (rest result))
       (dolist (x (rest result) nil)
         (print-warning x)))))

(defun monitor-act-r-command (act-r-command monitor &optional when)
  (let ((a (make-action :type (if (eq when :after) 'after-command (if (eq when :before) 'before-command 'simple-command))
                        :parameters (list act-r-command monitor)
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if (listp params)
          (values-list (cons success params))
        (values-list (list success params))))))


(defun remove-act-r-command-monitor (act-r-command monitor)
  (let ((a (make-action :type 'remove-monitor
                        :parameters (list act-r-command monitor)
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if (listp params)
          (values-list (cons success params))
        (values-list (list success params))))))

(defun list-act-r-commands ()
  (let ((a (make-action :type 'list-actions
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if success
          (values-list params)
        (dolist (x params nil)
         (print-warning x))))))

(defun check-act-r-command (name)
  (let ((a (make-action :type 'check
                        :parameters (list name)
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if success
          (values-list params)
        (dolist (x params nil)
         (print-warning x))))))


(defun list-act-r-connections ()
  (let ((a (make-action :type 'list-connections
                        :evaluator :lisp)))
    (multiple-value-bind (success params)
        (send-des-action a)
      (if success
          (values-list params)
        (dolist (x params nil)
         (print-warning x))))))

(defun return-result (handler a success result)
  (let (p)
    (bt:with-lock-held ((handler-received-requests-lock handler))
      (setf p (find a (handler-received-requests handler) :key 'pending-request-action))
      (when p (setf (handler-received-requests handler) (remove p (handler-received-requests handler)))))
    
    (if p
        (when (pending-request-id p)
          (let (res json-result)
            (handler-case
                (progn
                  (setf res (json:encode-json-to-string result))
                  (setf json-result (if success
                                        (format nil "{\"result\": ~a, \"error\": null, \"id\": ~s}~c" res (pending-request-id p) (code-char 4))
                                      (format nil "{\"result\": null, \"error\": {\"message\": ~a}, \"id\": ~s}~c" res (pending-request-id p) (code-char 4)))))
              (error ()
                (setf json-result (format nil "{\"result\": null, \"error\": {\"message\": ~s}, \"id\": ~s}~c" 
                                    (format nil "Could not convert result ~s to JSON" result) (pending-request-id p) (code-char 4)))))
            (handler-case
                (bt:with-lock-held ((handler-stream-lock handler))
                  (format (usocket:socket-stream (handler-socket handler)) "~a" json-result)
                  (force-output (usocket:socket-stream (handler-socket handler))))
              (error ()
                (let ((*print-circle* t)) (send-error-output "Unable to send return result for action ~s with parameters ~s." (action-type a) (action-parameters a)))))))
      (let ((*print-circle* t)) (send-error-output "Don't know how to return a result for action ~s with parameters ~s to handler connection ~s." (action-type a) (action-parameters a) (handler-socket handler))))))

(defvar *action-locks* 0)

#+:single-threaded-act-r 
(defun send-des-action (a)
  (handler-case 
      (progn
        (perform-action *dispatcher* a)
        (values (action-result-success a) (action-result a)))
    
    (error (x) (let* ((*print-circle* t)
                      (string (format nil "Error ~/print-error-message/ occurred while waiting for action ~s ~s." x (action-type a) (action-parameters a))))
                 (send-error-output string)
                 (values nil string)))))
  


#-:single-threaded-act-r (defun send-des-action (a)
  ;; details already filled in just create the internal controls
  (setf (action-cv a) (bt:make-condition-variable))
  (setf (action-lock a) (bt:make-lock ))
  
  (bt:acquire-lock (action-lock a) t)
  (add-action-to-queue a)

  (handler-case 
      (progn
        (loop
          (when (action-complete a)
            (return))
          (bt:condition-wait (action-cv a) (action-lock a)))
        (values (action-result-success a) (action-result a)))
    
    (error (x) (let* ((*print-circle* t)
                      (string (format nil "Error ~/print-error-message/ occurred while waiting for action ~s ~s." x (action-type a) (action-parameters a))))
                 (send-error-output string)
                 (values nil string)))
    (condition (x) (let* ((*print-circle* t)
                          (string (format nil "Condition ~/print-error-message/ occurred while waiting for action ~s ~s." x (action-type a) (action-parameters a))))
                     (send-error-output string)
                     (values nil string)))))

    
(defun add-action-to-queue (action)
  (bt:with-lock-held ((dispatcher-action-lock *dispatcher*))
    (push-last action (dispatcher-action-queue *dispatcher*))
    (bt:condition-notify (dispatcher-action-cv *dispatcher*))))


(defun dispatcher-process-actions (dispatcher)
  (bt:acquire-lock (dispatcher-action-lock dispatcher) t)
  (loop 
    (dolist (x (dispatcher-action-queue dispatcher))
      (handler-case
          (perform-action dispatcher x)
        (error (e) (let ((*print-circle* t)) (send-error-output "Error ~/print-error-message/ occurred during perform-action for action ~s ~s." e (action-type x) (action-parameters x))))
        (condition (c) (let ((*print-circle* t)) (send-error-output "Condition ~/print-error-message/ occurred during perform-action for action ~s ~s." c (action-type x) (action-parameters x))))))
    (setf (dispatcher-action-queue dispatcher) nil)
    (bt:condition-wait (dispatcher-action-cv dispatcher) (dispatcher-action-lock dispatcher))))



(defvar *server-host*)
(defvar *server-port*)

(defvar *force-local* nil)

#-:single-threaded-act-r
(defun start-des (&optional (create t) given-host (remote-port 2650))
  (let* ((host (if given-host
                   (progn
                     (setf *server-host* (ignore-errors (map 'vector 'parse-integer (usocket::split-sequence #\. given-host))))
                     given-host)
                 (let ((host-ip (unless *force-local*
                                  (ignore-errors (find-if (lambda (x) (and (= (length x) 4) (not (every 'zerop x))))
                                                          (usocket::get-hosts-by-name (usocket::get-host-name)))))))
                   (if host-ip
                       (progn
                         (setf *server-host* host-ip)
                         (usocket::vector-quad-to-dotted-quad host-ip))
                     (let ((local-host-ip (ignore-errors (find-if (lambda (x) (and (= (length x) 4) (not (every 'zerop x))))
                                                                  (usocket::get-hosts-by-name "localhost")))))
                       ;; when defaulting to localhost set the allow externals on
                       ;; because other connections from the same machine may
                       ;; show the 'real' address and a true external couldn't get
                       ;; in anyway.
                       (setf *allow-external-connections* t)
                       
                       (if local-host-ip
                           (progn
                             (setf *server-host* local-host-ip)
                             (usocket::vector-quad-to-dotted-quad local-host-ip))
                         (progn
                           (setf *server-host* #(127 0 0 1))
                           "127.0.0.1")))))))
         (server-socket (loop (format t "Trying to open server at ~s:~s~%" host remote-port)
                              (let ((s (handler-case (usocket:socket-listen host remote-port)
                                         (usocket:address-in-use-error () (progn
                                                                            (send-error-output "Could not open a socket for host ~s port ~s because it is already open~%" host remote-port)
                                                                            (incf remote-port) 
                                                                            :again))
                                         (error (x) (progn
                                                      (send-error-output "Error ~/print-error-message/ occurred while trying to open a socket for host ~s port ~s~%" x host remote-port)
                                                      nil)))))
                                (cond ((null s) 
                                       (format t "Server not started.~%")
                                       (return-from start-des nil))
                                      ((eq s :again))
                                      (t (format t "Server started on port ~s~%" remote-port)
                                         (return s)))))))
    (setf *server-port* remote-port)
    (if server-socket
        (progn
          (if create
              (setf *dispatcher*
                (make-dispatcher 
                 :connection-socket server-socket
                 :action-lock (bt:make-lock "dispatcher-action-lock") 
                 :handler-lock (bt:make-lock "dispatcher-handler-lock")
                 :command-lock (bt:make-lock "dispatcher-command-lock")
                 :action-cv (bt:make-condition-variable :name "dispatcher-action-condition-variable")
                 :available-received-lock (bt:make-lock "dispatcher-available-received-lock")
                 :available-execute-lock (bt:make-lock "dispatcher-available-execute-lock")))
            (setf (dispatcher-connection-socket *dispatcher*) server-socket))
          
          (setf (dispatcher-connection-interface *dispatcher*)
            (bt:make-thread (lambda ()
                              (let ((*package* *default-package*))
                                (loop
                                  (let ((new (usocket:socket-accept server-socket)))
                                    (initialize-dispatch-connection *dispatcher* new)))))
                            :name "initialize-dispatch-connection"))
          
          (setf (dispatcher-action-thread *dispatcher*)
            (bt:make-thread (lambda ()
                              (let ((*package* *default-package*))
                                (dispatcher-process-actions *dispatcher*)))
                            :name "dispatcher-process-actions"))
          
          (handler-case (with-open-file (f "~/act-r-address.txt" :direction :output :if-exists :supersede :if-does-not-exist :create)
                          (format f "~a" host))
            (error (x)
              (send-error-output "Error ~/print-error-message/ occurred while trying to write the address to ~s~%" x (translate-logical-pathname "~/act-r-address.txt"))))
          
          (handler-case (with-open-file (f "~/act-r-port-num.txt" :direction :output :if-exists :supersede :if-does-not-exist :create)
                          (format f "~d" remote-port))
            (error (x)
              (send-error-output "Error ~/print-error-message/ occurred while trying to write the port number to ~s~%" x (translate-logical-pathname "~/act-r-port-num.txt"))))
          
          (handler-case (with-open-file (f (translate-logical-pathname "ACT-R:environment;GUI;init;05-current-net.tcl") :direction :output :if-exists :supersede :if-does-not-exist :create)
                          (multiple-value-bind
                                (second minute hour date month year)
                              (get-decoded-time)
                            (format f "# Port settings for ACT-R server started at ~2,'0d:~2,'0d:~2,'0d ~d/~2,'0d/~d~%set actr_port ~d~%set actr_address \"~a\"~%"
                              hour minute second month date year remote-port host)))
            (error (x)
              (send-error-output "Error ~/print-error-message/ occurred while trying to write the Environment network config file ~s~%" x (translate-logical-pathname "ACT-R:environment;GUI;init;05-current-net.tcl"))))
          
          
          t)
      nil)))

#+:single-threaded-act-r
(defun start-des (&optional (create t) given-host (remote-port 2650))
  (declare (ignore create given-host remote-port))
  (format t "Single threaded ACT-R cannot start a dispatcher."))



(defun dont-start-des ()
  (setf *dispatcher*
    (make-dispatcher 
     :connection-socket nil
     :action-lock (bt:make-lock "dispatcher-action-lock") 
     :handler-lock (bt:make-lock "dispatcher-handler-lock")
     :command-lock (bt:make-lock "dispatcher-command-lock")
     :action-cv (bt:make-condition-variable :name "dispatcher-action-condition-variable")
     :available-received-lock (bt:make-lock "dispatcher-available-received-lock")
     :available-execute-lock (bt:make-lock "dispatcher-available-execute-lock")))
  (setf (dispatcher-action-thread *dispatcher*)
    #-:single-threaded-act-r (bt:make-thread (lambda ()
                                               (let ((*package* *default-package*))
                                                 (dispatcher-process-actions *dispatcher*)))
                                             :name "dispatcher-process-actions")
    #+:single-threaded-act-r nil))
          
#-:single-threaded-act-r
(defun init-des ()
  (start-des nil))

#-:single-threaded-act-r
(defun stop-des ()
  (when (usocket::usocket-p (dispatcher-connection-socket *dispatcher*))
    (usocket:socket-close (dispatcher-connection-socket *dispatcher*)))
  (when (and (bt:threadp (dispatcher-connection-interface *dispatcher*))
               (bt:thread-alive-p (dispatcher-connection-interface *dispatcher*)))
    (bt:destroy-thread (dispatcher-connection-interface *dispatcher*)))
  (when (and (bt:threadp (dispatcher-action-thread *dispatcher*))
               (bt:thread-alive-p (dispatcher-action-thread *dispatcher*)))
    (bt:destroy-thread (dispatcher-action-thread *dispatcher*)))
  (dolist (x (dispatcher-available-execute-workers *dispatcher*))
    (when (and (bt:threadp (worker-thread x))
               (bt:thread-alive-p (worker-thread x)))
      (bt:destroy-thread (worker-thread x))))
  (dolist (x (dispatcher-available-received-workers *dispatcher*))
    (when (and (bt:threadp (worker-thread x))
               (bt:thread-alive-p (worker-thread x)))
      (bt:destroy-thread (worker-thread x)))))




;;; Assuming JSON-RPC 1.0 communication at this point
;;; Char code 4 terminates a message in and out

(defun initialize-dispatch-connection (dispatcher socket)
  
  (if (and (null *allow-external-connections*)
           (not (or (usocket:ip= (usocket::get-peer-address socket) *server-host*)
                    (usocket:ip= (usocket::get-peer-address socket) #(172 17 0 2)))))
           
      (send-error-output "Attempted connection from ~s was denied~%" (usocket::vector-quad-to-dotted-quad (usocket::get-peer-address socket)))
    
    (let ((handler (make-handler :socket socket)))
      (setf (handler-thread handler) (bt:make-thread (lambda ()
                                                       (let ((*package* *default-package*))
                                                         (unwind-protect
                                                             (process-external-input handler (usocket:socket-stream socket))
                                                           
                                                           (handler-case 
                                                               (progn 
                                                                 
                                                                 ;; Things we've received and not initiated should be removed
                                                                 (dolist (x (handler-commands handler))
                                                                   (remove-command dispatcher (dispatch-command-name x)))
                                                                 
                                                                 ;; things we've sent but haven't received need to return an error result
                                                                 (bt:with-lock-held ((handler-sent-requests-lock handler))
                                                                   (dolist (x (handler-sent-requests handler))
                                                                     (abort-request x)))
                                                                 
                                                                 ;; things we've received but haven't returned should probably have their
                                                                 ;; threads killed, but for now just going to let them run out naturally
                                                                 ;; and fail when trying to return results
                                                                 
                                                                 (bt:with-lock-held ((dispatcher-handler-lock dispatcher))
                                                                   (setf (dispatcher-connections dispatcher)
                                                                     (remove handler (dispatcher-connections dispatcher))))
                                                                 
                                                                 (usocket:socket-close socket))
                                                             (error (x) (send-error-output "Error ~/print-error-message/ encountered while cleaning up a terminated connection." x))))))
                                                     :name "process-external-input"))
      (bt:with-lock-held ((dispatcher-handler-lock dispatcher))
        (push handler (dispatcher-connections dispatcher))))))

                                      
(defun received-worker (worker)
  
  (bt:acquire-lock (worker-lock worker) t)
  
  (bt:with-lock-held ((worker-start-lock worker))
    (bt:condition-notify (worker-started worker)))
  
  (loop
      (bt:condition-wait (worker-cv worker) (worker-lock worker))
      
      (multiple-value-bind (success result)
          (send-des-action (worker-action worker))
        (return-result (worker-handler worker) (worker-action worker) success result))
      
      (bt:with-lock-held ((dispatcher-available-received-lock *dispatcher*))
        (push-last worker (dispatcher-available-received-workers *dispatcher*)))))


(defun execute-worker (worker)
  
  (bt:acquire-lock (worker-lock worker) t)
  
  (bt:with-lock-held ((worker-start-lock worker))
    (bt:condition-notify (worker-started worker)))
  
  (loop
    (bt:condition-wait (worker-cv worker) (worker-lock worker))
    
    (let* ((*package* *default-package*)
           (others (worker-other worker))
           (c (first others))
           (parameters (second others)))
      
      (execute-command (worker-action worker)  c parameters))
    
    (bt:with-lock-held ((dispatcher-available-execute-lock *dispatcher*))
      (push-last worker (dispatcher-available-execute-workers *dispatcher*)))))

(defvar *r-w-num* 0)

(defun process-external-input (handler stream)
  (handler-case
      (let ((s (make-array 40
                           :element-type 'character
                           :adjustable T
                           :fill-pointer 0)))
        (loop
          (let ((char (handler-case (read-char stream nil :done)
                        (error ()
                          (send-error-output "External connection terminated while trying to read data.~%")
                          (return-from process-external-input)))))
            
            (cond ((eq char :done)
                   (send-error-output "External connection closed.")
                   (return))
                  ((eq (char-code char) 4)
                   (let ((message (handler-case (json:decode-json-from-string s)
                                    ((or error condition) (x)
                                     (send-error-output "Problem encountered decoding JSON string ~s: ~/print-error-message/. Connection terminated" s x)
                                     (return-from process-external-input nil)))))
                     
                     (if (= (length message) 3)
                         (let ((method (assoc :method message :test 'string=))
                               (result (assoc :result message :test 'string=)))
                           
                           (cond (method
                                      
                                  (let* ((m (cdr method))
                                         (p (assoc :params message :test 'string=))
                                         (i (assoc :id message :test 'string=)))
                                    (unless (and p i)
                                      (send-error-output "Invalid message encountered decoding JSON string ~s. Connection terminated" s)
                                      (return-from process-external-input nil))
                                    (let* ((params (cdr p))
                                           (id (cdr i))
                                           (action (cond 
                                                    ((and (string-equal m "add")
                                                          (>= (length params) 2)
                                                          (stringp (first params))
                                                          (or (null (second params)) (stringp (second params)))
                                                          (or (null (third params)) (stringp (third params)))
                                                          (or (null (fifth params)) (stringp (fifth params))))
                                                     (make-action :type 'add-command
                                                                  :parameters (list :name (first params)
                                                                                    :underlying-function (second params)
                                                                                    :documentation (third params)
                                                                                    :evaluator handler
                                                                                    :single-instance (fourth params)
                                                                                    :local-name (fifth params))))
                                                    ((and (string-equal m "remove")
                                                          (= (length params) 1)
                                                          (stringp (first params)))
                                                     (make-action :type 'remove-command
                                                                  :parameters (list (first params))
                                                                  :evaluator handler))
                                                    ((and (string-equal m "evaluate")
                                                          (>= (length params) 1)
                                                          (stringp (first params)))
                                                     (make-action :type 'execute-command
                                                                  :model (second params)
                                                                  :parameters (list :name (first params) 
                                                                                    :parameters (cddr params))
                                                                  :evaluator handler))
                                                    ((and (string-equal m "monitor")
                                                          (>= (length params) 2)
                                                          (stringp (first params))
                                                          (stringp (second params)))
                                                     (make-action :type (if (and (stringp (third params)) (string-equal (third params) "after")) 
                                                                            'after-command 
                                                                          (if (and (stringp (third params)) (string-equal (third params) "before"))
                                                                              'before-command
                                                                            'simple-command))
                                                                  :parameters (list (first params) (second params))
                                                                  :evaluator handler))
                                                    ((and (string-equal m "remove-monitor")
                                                          (= (length params) 2)
                                                          (stringp (first params))
                                                          (stringp (second params)))
                                                     (make-action :type 'remove-monitor
                                                                  :parameters params
                                                                  :evaluator handler))
                                                    ((and (string-equal m "list-commands")
                                                          (null params))
                                                     (make-action :type 'list-actions
                                                                  :evaluator handler))
                                                    ((and (string-equal m "list-connections")
                                                          (null params))
                                                     (make-action :type 'list-connections
                                                                  :evaluator handler))
                                                    ((and (string-equal m "check")
                                                          (= (length params) 1)
                                                          (stringp (first params)))
                                                     (make-action :type 'check
                                                                  :parameters params
                                                                  :evaluator handler))
                                                    ((and (string-equal m "set-name")
                                                          (= (length params) 1)
                                                          (stringp (first params)))
                                                     (make-action :type 'set-name
                                                                  :parameters params
                                                                  :evaluator handler)))))
                                      (if action
                                          (let ((p (make-pending-request :action action :id id)))
                                            (bt:with-lock-held ((handler-received-requests-lock handler))
                                              (push-last p (handler-received-requests handler)))
                                            
                                            (let ((worker nil))
                                              
                                              (bt:with-lock-held ((dispatcher-available-received-lock *dispatcher*))
                                                (if (dispatcher-available-received-workers *dispatcher*)
                                                    (progn
                                                      (setf worker (pop (dispatcher-available-received-workers *dispatcher*)))
                                                      (setf (worker-handler worker) handler)
                                                      (setf (worker-action worker) action))
                                                  (progn
                                                    (setf worker (make-worker
                                                                  :handler handler
                                                                  :action action
                                                                  :lock (bt:make-lock )
                                                                  :cv (bt:make-condition-variable)
                                                                  :started (bt:make-condition-variable)
                                                                  :start-lock (bt:make-lock)))
                                                    
                                                    (bt:acquire-lock (worker-start-lock worker))
                                                    (setf (worker-thread worker)
                                                      (bt:make-thread (lambda () (received-worker worker))
                                                                      :name (concatenate 'string "received-worker " (princ-to-string *r-w-num*))))
                                                    (bt:condition-wait (worker-started worker) (worker-start-lock worker))
                                                    (bt:release-lock (worker-start-lock worker))))
                                                
                                                (bt:with-lock-held ((worker-lock worker))
                                                  (bt:condition-notify (worker-cv worker))))))
                                        (progn
                                          (send-error-output "Invalid message ~s." s)
                                          (let* ((a (make-action :type 'invalid))
                                                 (p (make-pending-request :action a :id id)))
                                    
                                            (bt:with-lock-held ((handler-received-requests-lock handler))
                                              (push-last p (handler-received-requests handler)))
                                            (return-result handler a nil (cond ((find m  (list "add" "remove" "evaluate" "monitor" "remove-monitor" "check" "list-commands" "set-name" "list-connections") :test 'string-equal)
                                                                                (format nil "Invalid form for ~s method: ~s" m s))
                                                                               (t (format nil "Invalid method ~s. Only add, remove, evaluate, monitor, remove-monitor, check, set-name, list-connections, and list-commands are allowed." m))))))))))
                                 (result
                                  
                                  (let* ((r (cdr result))
                                         (e (assoc :error message :test 'string=))
                                         (i (assoc :id message :test 'string=)))
                                    (unless (and e i)
                                      (send-error-output "Invalid message encountered decoding JSON string ~s. Connection terminated" s)
                                      (return-from process-external-input nil))
                                    (let ((id (cdr i))
                                          (error (cdr e)))
                                      (let (p)
                                        (bt:with-lock-held ((handler-sent-requests-lock handler))
                                          (setf p (find id (handler-sent-requests handler) :key 'pending-request-id :test 'equalp))
                                          (when p (setf (handler-sent-requests handler) (remove p (handler-sent-requests handler)))))
                                        
                                        (if p
                                            (progn ;let ((a (pending-request-action p)))
                                              (bt:acquire-lock (pending-request-lock p))
                                              (setf (pending-request-success p) (if r t nil))
                                              (setf (pending-request-result p) (if r r 
                                                                                 (let ((messages (cdr (assoc :message error))))
                                                                                   (if messages
                                                                                       messages
                                                                                     (list "No error messages received.")))))
                                              (setf (pending-request-complete p) t)
                                              (bt:condition-notify (pending-request-cv p))
                                              (bt:release-lock (pending-request-lock p)))
                                          (progn
                                            (send-error-output "Returned result has an id that does not match a pending request ~s" s)
                                            (return-from process-external-input nil)))))))
                                 (t
                                  (send-error-output "Invalid message received ~s terminating connection" s)
                                  (return-from process-external-input nil))))
                       (progn
                         (send-error-output "Invalid message received ~s terminating connection" s)
                         (return-from process-external-input nil))))
                       (setf (fill-pointer s) 0))
                  (t (vector-push-extend char s))))))
    ((or error condition) (x)
     (send-error-output "Error ~/print-error-message/ occurred in process-external-input for stream ~s.  Connection terminated." x stream))))
                      

  
  

(defun remove-command (dispatcher name)
  (bt:with-lock-held ((dispatcher-command-lock dispatcher))
    (let ((current (gethash name (dispatcher-command-table dispatcher))))
      (if current
          (progn
            (remhash name (dispatcher-command-table dispatcher))
            
            ;; Take this command off all the commands it is monitoring
            (dolist (x (dispatch-command-monitoring current))
              (setf (dispatch-command-before x) (remove current (dispatch-command-before x)))
              (setf (dispatch-command-after x) (remove current (dispatch-command-after x)))
              (setf (dispatch-command-simple x) (remove current (dispatch-command-simple x))))
            
            ;; Remove this command from those that are monitoring it
            (dolist (x (dispatch-command-before current))
              (setf (dispatch-command-monitoring x) (remove current (dispatch-command-monitoring x))))
            (dolist (x (dispatch-command-after current))
              (setf (dispatch-command-monitoring x) (remove current (dispatch-command-monitoring x))))
            (dolist (x (dispatch-command-simple current))
              (setf (dispatch-command-monitoring x) (remove current (dispatch-command-monitoring x))))
            
            (let ((h (dispatch-command-evaluator current)))
              (unless (eq :lisp h) 
                (bt:with-lock-held ((handler-command-lock h))
                  (setf (handler-commands h) (remove current (handler-commands h))))))
            
            t)         
        (format nil "Command ~s not in the table and cannot be removed." name)))))


(defun abort-request (handler-request)
  ;; set the action details to indicate failure
    (bt:with-lock-held ((pending-request-lock handler-request)) 
      (setf (pending-request-success handler-request) nil)
      (setf (pending-request-result handler-request) "Connection to command owner lost.")
      (setf (pending-request-complete handler-request) t)
    ;; just notify the request so that it returns the result
      (bt:condition-notify (pending-request-cv handler-request))))
  
;(defvar *a* nil)

(defvar *e-w-num* 0)


;; There is probably a better way to handle this, but I've run into problems with
;; trying to build a macro to call defmacro since it is a macro, and other
;; attempts have been only partially successful.
;;
;; The fmakunbound avoids a warning in some Lisps, but because LispWorks still
;; wants to signal a redefinition (for something that is undefined) need to 
;; build a handler for muffle-warning.

(defun quiet-warning-handler (c)
  (let ((r (find-restart 'muffle-warning c)))
    (when r 
      (invoke-restart r))))

(defun create-local-macro (name cmd)
  (let ((*read-eval* t)
        (fn (string->name name)))
    (fmakunbound fn)
    (handler-bind ((warning #'quiet-warning-handler)) 
      (read-from-string (format nil "#.(defmacro ~:@(~a~) (&rest r) `(dispatch-apply-list ~s ',r))" name cmd)))))


#| works in ACL, CCL, and SBCL, but not LispWorks because fmakeunbound doesn't
   suppress their redifintion warning that throws a condition which gets handled
   elsewhere and thus prevents the completion of the defmethod.

(defun create-local-macro (name cmd)
  (ignore-errors
   (let ((*read-eval* t)
         (fn (string->name name)))
     
     (fmakunbound fn)
     (read-from-string (format nil "#.(defmacro ~:@(~a~) (&rest r) `(dispatch-apply-list ~s ',r))" name cmd)))))
|#


#|

This works in ACL, once in LispWorks and SBCL (probably because of a condition which isn't trapped now),
but not at all in CCL.

(defmacro create-the-macro (cmd)
  `(defmacro internal-dummy-macro (&rest r) `(dispatch-apply-list ,,cmd ',r)))


(defun create-local-macro (name cmd)
  (let ((n (string->name name)))
    (create-the-macro cmd)
    (setf (macro-function n) (macro-function 'internal-dummy-macro))))
|#

#|

This almost works, but throws an error in ACL from the eval 
(defmacro c-t-m (n cmd)
  ``(defmacro ,,n (&rest r) `(dispatch-apply-list ,,,cmd ',r)))


(defun create-local-macro (name cmd)
  (let ((n (string->name name)))
    (eval (c-t-m n cmd))))
|#


#-:single-threaded-act-r (defun perform-action (dispatcher action)
  (case (action-type action)
    (add-command 
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action) t)
           (handler-case
               (destructuring-bind (&key name documentation underlying-function evaluator single-instance local-name) (action-parameters action)
                 (declare (ignorable documentation single-instance))
                 (if (and (stringp name)
                          (or (null underlying-function)
                              (and (eq :lisp evaluator) (or (functionp underlying-function)
                                                            (and (symbolp underlying-function)
                                                                 (fboundp underlying-function))))
                              (and (not (eq :lisp evaluator)) (stringp underlying-function))))
                     (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                       (let ((current (gethash name (dispatcher-command-table dispatcher))))
                         (if current
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s already exists in the table and cannot be added." name))
                           (let ((c (apply 'make-dispatch-command (action-parameters action))))
                             (setf (gethash name (dispatcher-command-table dispatcher)) c)
                             ;; a new-name has been used
                             (remhash name (dispatcher-spec-names dispatcher))
                             (let ((h (dispatch-command-evaluator c)))
                               (unless (eq :lisp h) 
                                 (bt:with-lock-held ((handler-command-lock h))
                                   (push c (handler-commands h)))))
                             
                             (when (stringp local-name)
                               (create-local-macro local-name name))
                             
                             (setf (action-result-success action) t
                               (action-result action) (list name))))
                         (setf (action-complete action) t)))
                   (progn
                     (setf (action-result-success action) nil)
                     (setf (action-result action) (format nil "Invalid parameters when trying to add command specified with name ~s and function ~s" name underlying-function))
                     (setf (action-complete action) t))))
             ((or error condition) (x) 
              (let ((*print-circle* t))
                (setf (action-result-success action) nil)
                (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to add command defined by ~s" x (action-parameters action)))
                (setf (action-complete action) t)))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (remove-command 
     (unwind-protect
         (progn (bt:acquire-lock (action-lock action))
           (handler-case
               (destructuring-bind (name) (action-parameters action)
                 (unless (stringp name)
                   ;; should it only allow the owner to remove a command or
                   ;; should anyone be allowed? Letting anyone allows someone to 
                   ;; override a provided command, but is that a good idea?
                   (error "Invalid remove name")) 
                 (let ((r (remove-command dispatcher name)))
                   (if (stringp r)
                       (progn 
                         (setf (action-result-success action) nil)
                         (setf (action-result action) r))
                     (progn
                       (setf (action-result-success action) t)
                       (setf (action-result action) (list name))))
                   (setf (action-complete action) t)))
             
             ((or error condition) (x) 
              (let ((*print-circle* t))
                (setf (action-result-success action) nil)
                (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to remove command specified by ~s" x (action-parameters action)))
                (setf (action-complete action) t)))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    
    (execute-command 
       (handler-case
         (destructuring-bind (&key name parameters) (action-parameters action)
           (unless (stringp name)
                   ;; should there be anything else checked?
             (error "Invalid execute name"))
           (bt:with-lock-held ((dispatcher-command-lock dispatcher))
             (let ((c (gethash name (dispatcher-command-table dispatcher))))
               (cond (c
                      (let ((worker nil))
                        
                        (bt:with-lock-held ((dispatcher-available-execute-lock *dispatcher*))
                          (if (dispatcher-available-execute-workers *dispatcher*)
                              (progn
                                (setf worker (pop (dispatcher-available-execute-workers *dispatcher*)))
                                (setf (worker-action worker) action)
                                (setf (worker-other worker) (list c parameters)))
                            (progn
                              (setf worker (make-worker
                                            :action action
                                            :other (list c parameters)
                                            :lock (bt:make-lock )
                                            :cv (bt:make-condition-variable )
                                            :started (bt:make-condition-variable)
                                            :start-lock (bt:make-lock )))
                              (bt:acquire-lock (worker-start-lock worker))
                              
                              (setf (worker-thread worker)
                                (bt:make-thread (lambda () (execute-worker worker))
                                                :name (concatenate 'string "execute-worker " (princ-to-string *e-w-num*))))
                              (bt:condition-wait (worker-started worker) (worker-start-lock worker))
                              (bt:release-lock (worker-start-lock worker))
                              ))
                          (bt:with-lock-held ((worker-lock worker))
                            (bt:condition-notify (worker-cv worker))))))
                     (t
                      (bt:with-lock-held ((action-lock action))
                        (setf (action-result-success action) nil
                          (action-result action) (format nil "Command ~s not found in the table and cannot be executed." name)
                          (action-complete action) t)
                        (bt:condition-notify (action-cv action))))))))
       ((or error condition) (x) 
        (bt:with-lock-held ((action-lock action))
          (let ((*print-circle* t))
            (setf (action-result-success action) nil)
            (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to execute command specified by ~s" x (action-parameters action)))
            (setf (action-complete action) t)
            (bt:condition-notify (action-cv action)))))))
    
    
    (before-command      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (destructuring-bind (command-name before-name) (action-parameters action)
                 (unless (and (stringp command-name) (stringp before-name))
                   (error "Invalid before monitoring parameter"))                 
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let ((current (gethash before-name (dispatcher-command-table dispatcher))))
                     (if current
                         (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                           (if c
                               (if (find current (dispatch-command-before c))
                                   (setf (action-result-success action) nil
                                     (action-result action) (format nil "Command ~s already on the before list for command ~s." before-name command-name))
                                 (progn
                                   (push-last current (dispatch-command-before c))
                                   (pushnew c (dispatch-command-monitoring current))
                                   (setf (action-result-success action) t
                                     (action-result action) (list before-name))))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s does not exist so command ~s cannot be called before it." command-name before-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist thus it cannot be called before another command." before-name)))
                     (setf (action-complete action) t))))
             ((or error condition) (x) 
              (let ((*print-circle* t))
                (setf (action-result-success action) nil)
                (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
                (setf (action-complete action) t)))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
     
    (after-command      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (destructuring-bind (command-name after-name) (action-parameters action)
                 (unless (and (stringp command-name) (stringp after-name))
                   (error "Invalid after monitoring parameter"))
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let ((current (gethash after-name (dispatcher-command-table dispatcher))))
                     (if current
                         (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                           (if c
                               (if (find current (dispatch-command-after c))
                                   (setf (action-result-success action) nil
                                     (action-result action) (format nil "Command ~s already on the after list for command ~s." after-name command-name))
                                 (progn
                                   (push-last current (dispatch-command-after c))
                                   (pushnew c (dispatch-command-monitoring current))
                                   (setf (action-result-success action) t
                                     (action-result action) (list after-name))))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s does not exist so command ~s cannot be called after it." command-name after-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist thus it cannot be called after another command." after-name)))
                     (setf (action-complete action) t))))
             ((or error condition) (x) 
              (let ((*print-circle* t))
                (setf (action-result-success action) nil)
                (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
                (setf (action-complete action) t)))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (simple-command      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (destructuring-bind (command-name after-name) (action-parameters action)
                 (unless (and (stringp command-name) (stringp after-name))
                   (error "Invalid simple monitoring parameter"))
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let ((current (gethash after-name (dispatcher-command-table dispatcher))))
                     (if current
                         (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                           (if c
                               (if (find current (dispatch-command-simple c))
                                   (setf (action-result-success action) nil
                                     (action-result action) (format nil "Command ~s already on the after list for command ~s." after-name command-name))
                                 (progn
                                   (push-last current (dispatch-command-simple c))
                                   (pushnew c (dispatch-command-monitoring current))
                                   (setf (action-result-success action) t
                                     (action-result action) (list after-name))))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s does not exist so command ~s cannot be called after it." command-name after-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist thus it cannot be called after another command." after-name)))
                     (setf (action-complete action) t))))
             ((or error condition) (x) 
              (let ((*print-circle* t))
                (setf (action-result-success action) nil)
                (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
                (setf (action-complete action) t)))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (remove-monitor      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (destructuring-bind (command-name monitor) (action-parameters action)
                 (unless (and (stringp command-name) (stringp monitor))
                   ;; should it verify somehow that the connection that added the
                   ;; monitor be the one to remove it?
                   (error "Invalid remove monitor parameter"))
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                     (if c
                         (let ((m (gethash monitor (dispatcher-command-table dispatcher))))
                           (if m
                               (progn
                                 (setf (dispatch-command-before c) (remove m (dispatch-command-before c)))
                                 (setf (dispatch-command-after c) (remove m (dispatch-command-after c)))
                                 (setf (dispatch-command-simple c) (remove m (dispatch-command-simple c)))
                                 (setf (dispatch-command-monitoring m) (remove c (dispatch-command-monitoring m)))
                                 (setf (action-result-success action) t
                                   (action-result action) (list monitor)))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s does not exist so it cannot be removed as a monitor of ~s." monitor command-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist so monitor ~s does not need to be removed." command-name monitor)))))
                   
                   (setf (action-complete action) t))
             ((or error condition) (x) 
              (let ((*print-circle* t))
                (setf (action-result-success action) nil)
                (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
                (setf (action-complete action) t)))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (list-actions      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (progn
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let ((d nil))
                     (maphash (lambda (key value)
                                (push (list key (dispatch-command-documentation value)) d))
                              (dispatcher-command-table dispatcher))
                     (setf (action-result-success action) t
                       (action-result action) (list d))))                  
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to list ACT-R commands" x))
              (setf (action-complete action) t))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (check      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (progn
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let* ((name (first (action-parameters action)))
                          (entry (gethash name (dispatcher-command-table dispatcher)))
                          (speculative (gethash name (dispatcher-spec-names dispatcher))))
                     (if entry
                         (setf (action-result action) (list t (eq (action-evaluator action) (dispatch-command-evaluator entry)) (dispatch-command-documentation entry)))
                       (if speculative
                           (setf (action-result action) (list t nil "Reserved"))
                         (setf (action-result action) (list nil))))
                     (setf (action-result-success action) t)))                  
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to check ACT-R commands" x))
              (setf (action-complete action) t))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (set-name      
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (let ((name (first (action-parameters action)))
                     (handler (action-evaluator action)))
                 (if (and name handler)
                     (progn
                       (bt:with-lock-held ((handler-command-lock handler))
                         (setf (handler-name handler) name))
                       
                       (setf (action-result action) 
                         (list name)))
                   (setf (action-result action) (list nil)))
                 (setf (action-result-success action) t)
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to set connection name" x))
              (setf (action-complete action) t))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (list-connections
     (unwind-protect
         (progn
           (bt:acquire-lock (action-lock action))
           (handler-case
               (let ((items (append (mapcar (lambda  (x)
                                              (list (bt:with-lock-held ((handler-command-lock x))
                                                      (handler-name x))
                                                    (bt:with-lock-held ((handler-stream-lock x))
                                                      (format nil "~{~d~^.~}:~d" (coerce (usocket::get-peer-address (handler-socket x)) 'list)
                                                        (usocket::get-peer-port (handler-socket x))))
                                                    (bt:with-lock-held ((handler-received-requests-lock x))
                                                      (length (handler-received-requests x)))
                                                    (bt:with-lock-held ((handler-sent-requests-lock x))
                                                      (length (handler-sent-requests x))) 
                                                    (sort (bt:with-lock-held ((handler-command-lock x))
                                                            (mapcar 'dispatch-command-name (handler-commands x)))
                                                          'string-lessp)))
                                      (bt:with-lock-held ((dispatcher-handler-lock dispatcher)) (dispatcher-connections dispatcher)))
                                    (list 
                                     (list "internal ACT-R commands" 
                                           (format nil "~{~d~^.~}:~d" (coerce *server-host* 'list)
                                             *server-port*)
                                           -1 -1 
                                           (let ((cmds nil))
                                             (maphash (lambda (key value)
                                                        (when (eq (dispatch-command-evaluator value) :lisp)
                                                          (push key cmds)))
                                                      (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                                                        (dispatcher-command-table dispatcher)))
                                             (sort cmds 'string-lessp)))))))
                 (setf (action-result-success action) t
                   (action-result action) (list items))
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to list ACT-R connections" x))
              (setf (action-complete action) t))))
       (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))
    
    (t 
     (bt:with-lock-held ((action-lock action))
       (setf (action-result-success action) nil)
       (setf (action-result action) (format nil "Invalid action type ~s." (action-type action)))
       (setf (action-complete action) t)
       (bt:condition-notify (action-cv action))))))



#+:single-threaded-act-r 
(defun perform-action (dispatcher action)
  (case (action-type action)
    (add-command 
     
     (handler-case
         (destructuring-bind (&key name documentation underlying-function evaluator single-instance local-name) (action-parameters action)
           (declare (ignorable documentation single-instance))
           (if (and (stringp name)
                    (or (null underlying-function)
                        (and (eq :lisp evaluator) (or (functionp underlying-function)
                                                      (and (symbolp underlying-function)
                                                           (fboundp underlying-function))))
                        ))
               (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                 (let ((current (gethash name (dispatcher-command-table dispatcher))))
                   (if current
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s already exists in the table and cannot be added." name))
                     (let ((c (apply 'make-dispatch-command (action-parameters action))))
                       (setf (gethash name (dispatcher-command-table dispatcher)) c)
                       ;; a new-name has been used
                       (remhash name (dispatcher-spec-names dispatcher))
                       (let ((h (dispatch-command-evaluator c)))
                         (unless (eq :lisp h) 
                           (bt:with-lock-held ((handler-command-lock h))
                             (push c (handler-commands h)))))
                       
                       (when (stringp local-name)
                         (create-local-macro local-name name))
                       
                       (setf (action-result-success action) t
                         (action-result action) (list name))))
                   (setf (action-complete action) t)))
             (progn
               (setf (action-result-success action) nil)
               (setf (action-result action) (format nil "Invalid parameters when trying to add command specified with name ~s and function ~s" name underlying-function))
               (setf (action-complete action) t))))
       ((or error condition) (x) 
        (let ((*print-circle* t))
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to add command defined by ~s" x (action-parameters action)))
          (setf (action-complete action) t)))))
    
    (remove-command 
     (handler-case
         (destructuring-bind (name) (action-parameters action)
           (unless (stringp name)
             ;; should it only allow the owner to remove a command or
             ;; should anyone be allowed? Letting anyone allows someone to 
             ;; override a provided command, but is that a good idea?
             (error "Invalid remove name")) 
           (let ((r (remove-command dispatcher name)))
             (if (stringp r)
                 (progn 
                   (setf (action-result-success action) nil)
                   (setf (action-result action) r))
               (progn
                 (setf (action-result-success action) t)
                 (setf (action-result action) (list name))))
             (setf (action-complete action) t)))
             
       ((or error condition) (x) 
        (let ((*print-circle* t))
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to remove command specified by ~s" x (action-parameters action)))
          (setf (action-complete action) t)))))
    
    (execute-command 
     (handler-case
         (destructuring-bind (&key name parameters) (action-parameters action)
           (unless (stringp name)
             ;; should there be anything else checked?
             (error "Invalid execute name"))
           (bt:with-lock-held ((dispatcher-command-lock dispatcher))
             (let ((c (gethash name (dispatcher-command-table dispatcher))))
               (cond (c
                      
                      (execute-command action c parameters))
                     (t
                      (setf (action-result-success action) nil
                        (action-result action) (format nil "Command ~s not found in the table and cannot be executed." name)
                        (action-complete action) t))))))
       ((or error ) (x) ;; don't catch conditions here to avoid loading problems but maybe better done elsewhere?
         (let ((*print-circle* t))
           (setf (action-result-success action) nil)
           (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to execute command specified by ~s" x (action-parameters action)))
           (setf (action-complete action) t)))))
    
    
    (before-command      
     (handler-case
         (destructuring-bind (command-name before-name) (action-parameters action)
           (unless (and (stringp command-name) (stringp before-name))
             (error "Invalid before monitoring parameter"))                 
           (bt:with-lock-held ((dispatcher-command-lock dispatcher))
             (let ((current (gethash before-name (dispatcher-command-table dispatcher))))
               (if current
                   (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                     (if c
                         (if (find current (dispatch-command-before c))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s already on the before list for command ~s." before-name command-name))
                           (progn
                             (push-last current (dispatch-command-before c))
                             (pushnew c (dispatch-command-monitoring current))
                             (setf (action-result-success action) t
                               (action-result action) (list before-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist so command ~s cannot be called before it." command-name before-name))))
                 (setf (action-result-success action) nil
                   (action-result action) (format nil "Command ~s does not exist thus it cannot be called before another command." before-name)))
               (setf (action-complete action) t))))
       ((or error condition) (x) 
        (let ((*print-circle* t))
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
          (setf (action-complete action) t)))))
       
     
    (after-command      
     (handler-case
         (destructuring-bind (command-name after-name) (action-parameters action)
           (unless (and (stringp command-name) (stringp after-name))
             (error "Invalid after monitoring parameter"))
           (bt:with-lock-held ((dispatcher-command-lock dispatcher))
             (let ((current (gethash after-name (dispatcher-command-table dispatcher))))
               (if current
                   (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                     (if c
                         (if (find current (dispatch-command-after c))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s already on the after list for command ~s." after-name command-name))
                           (progn
                             (push-last current (dispatch-command-after c))
                             (pushnew c (dispatch-command-monitoring current))
                             (setf (action-result-success action) t
                               (action-result action) (list after-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist so command ~s cannot be called after it." command-name after-name))))
                 (setf (action-result-success action) nil
                   (action-result action) (format nil "Command ~s does not exist thus it cannot be called after another command." after-name)))
               (setf (action-complete action) t))))
       ((or error condition) (x) 
        (let ((*print-circle* t))
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
          (setf (action-complete action) t)))))
    
    (simple-command      
     (handler-case
         (destructuring-bind (command-name after-name) (action-parameters action)
           (unless (and (stringp command-name) (stringp after-name))
             (error "Invalid simple monitoring parameter"))
           (bt:with-lock-held ((dispatcher-command-lock dispatcher))
             (let ((current (gethash after-name (dispatcher-command-table dispatcher))))
               (if current
                   (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
                     (if c
                         (if (find current (dispatch-command-simple c))
                             (setf (action-result-success action) nil
                               (action-result action) (format nil "Command ~s already on the after list for command ~s." after-name command-name))
                           (progn
                             (push-last current (dispatch-command-simple c))
                             (pushnew c (dispatch-command-monitoring current))
                             (setf (action-result-success action) t
                               (action-result action) (list after-name))))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist so command ~s cannot be called after it." command-name after-name))))
                 (setf (action-result-success action) nil
                   (action-result action) (format nil "Command ~s does not exist thus it cannot be called after another command." after-name)))
               (setf (action-complete action) t))))
       ((or error condition) (x) 
        (let ((*print-circle* t))
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
          (setf (action-complete action) t)))))
    
    (remove-monitor      
     (handler-case
         (destructuring-bind (command-name monitor) (action-parameters action)
           (unless (and (stringp command-name) (stringp monitor))
             ;; should it verify somehow that the connection that added the
             ;; monitor be the one to remove it?
             (error "Invalid remove monitor parameter"))
           (bt:with-lock-held ((dispatcher-command-lock dispatcher))
             (let ((c (gethash command-name (dispatcher-command-table dispatcher))))
               (if c
                   (let ((m (gethash monitor (dispatcher-command-table dispatcher))))
                     (if m
                         (progn
                           (setf (dispatch-command-before c) (remove m (dispatch-command-before c)))
                           (setf (dispatch-command-after c) (remove m (dispatch-command-after c)))
                           (setf (dispatch-command-simple c) (remove m (dispatch-command-simple c)))
                           (setf (dispatch-command-monitoring m) (remove c (dispatch-command-monitoring m)))
                           (setf (action-result-success action) t
                             (action-result action) (list monitor)))
                       (setf (action-result-success action) nil
                         (action-result action) (format nil "Command ~s does not exist so it cannot be removed as a monitor of ~s." monitor command-name))))
                 (setf (action-result-success action) nil
                   (action-result action) (format nil "Command ~s does not exist so monitor ~s does not need to be removed." command-name monitor)))))
           
           (setf (action-complete action) t))
       ((or error condition) (x) 
        (let ((*print-circle* t))
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to monitor command ~s" x (action-parameters action)))
          (setf (action-complete action) t)))))
    
    (list-actions      
     (handler-case
               (progn
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let ((d nil))
                     (maphash (lambda (key value)
                                (push (list key (dispatch-command-documentation value)) d))
                              (dispatcher-command-table dispatcher))
                     (setf (action-result-success action) t
                       (action-result action) (list d))))                  
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to list ACT-R commands" x))
              (setf (action-complete action) t))))
    
    (check      
     (handler-case
               (progn
                 (bt:with-lock-held ((dispatcher-command-lock dispatcher))
                   (let* ((name (first (action-parameters action)))
                          (entry (gethash name (dispatcher-command-table dispatcher)))
                          (speculative (gethash name (dispatcher-spec-names dispatcher))))
                     (if entry
                         (setf (action-result action) (list t (eq (action-evaluator action) (dispatch-command-evaluator entry)) (dispatch-command-documentation entry)))
                       (if speculative
                           (setf (action-result action) (list t nil "Reserved"))
                         (setf (action-result action) (list nil))))
                     (setf (action-result-success action) t)))                  
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to check ACT-R commands" x))
              (setf (action-complete action) t))))
    
    (set-name      
     (handler-case
               (let ((name (first (action-parameters action)))
                     (handler (action-evaluator action)))
                 (if (and name handler)
                     (progn
                       (bt:with-lock-held ((handler-command-lock handler))
                         (setf (handler-name handler) name))
                       
                       (setf (action-result action) 
                         (list name)))
                   (setf (action-result action) (list nil)))
                 (setf (action-result-success action) t)
                 (setf (action-complete action) t))
             ((or error condition) (x) 
              (setf (action-result-success action) nil)
              (setf (action-result action) (format nil "Error ~/print-error-message/ occurred while trying to set connection name" x))
              (setf (action-complete action) t))))
    
    (list-connections
     (setf (action-result-success action) t
       (action-result action) nil
       (action-complete action) t))
    
    (t 
       (setf (action-result-success action) nil)
       (setf (action-result action) (format nil "Invalid action type ~s." (action-type action)))
       (setf (action-complete action) t))))


#+:single-threaded-act-r 
(defun execute-command (action c parameters)
      (handler-case
          (progn
            (dolist (x (dispatch-command-before c))
              (evaluate-command action x (list (dispatch-command-name c) parameters nil nil)))
            (let ((result (if (dispatch-command-underlying-function c) (multiple-value-list (evaluate-command action c parameters)) (list t t))))
              (dolist (x (dispatch-command-after c))
                (evaluate-command action x (list (dispatch-command-name c) parameters (if (car result) t nil)
                                                 (if (car result) (cdr result) nil))))
              (dolist (x (dispatch-command-simple c))
                (evaluate-command action x parameters))
                
              (setf (action-result-success action) (car result))
              (setf (action-result action) (if (car result) (cdr result) (cadr result)))
              (setf (action-complete action) t)))
        (error (x) 
          (setf (action-result-success action) nil)
          (setf (action-result action) (format nil "Error ~/print-error-message/ occurred during execute-command for command ~s and parameters ~s" x (dispatch-command-name c) parameters))
          (setf (action-complete action) t))))

#-:single-threaded-act-r 
(defun execute-command (action c parameters)
  (unwind-protect
      (handler-case
          (progn
            (bt:acquire-lock (action-lock action))
            
            (when (dispatch-command-single-instance c)
              (if (stringp (dispatch-command-single-instance c))
                  (when (null (bt:acquire-lock (dispatch-command-lock c) nil))
                    (setf (action-result-success action) nil)
                    (setf (action-result action) (dispatch-command-single-instance c))
                    (setf (action-complete action) t)
                    (return-from execute-command))
                (bt:acquire-lock (dispatch-command-lock c) t)))
            
              (dolist (x (dispatch-command-before c))
                (evaluate-command action x (list (dispatch-command-name c) parameters nil nil)))
              (let ((result (if (dispatch-command-underlying-function c) (multiple-value-list (evaluate-command action c parameters)) (list t t))))
                (dolist (x (dispatch-command-after c))
                  (evaluate-command action x (list (dispatch-command-name c) parameters (if (car result) t nil)
                                            (if (car result) (cdr result) nil))))
                (dolist (x (dispatch-command-simple c))
                  (evaluate-command action x parameters))
                
                (setf (action-result-success action) (car result))
                (setf (action-result action) (if (car result) (cdr result) (cadr result)))
                (setf (action-complete action) t))
              (when (dispatch-command-single-instance c)
                (bt:release-lock (dispatch-command-lock c))))
        (error (x) 
         (when (dispatch-command-single-instance c)
           (bt:release-lock (dispatch-command-lock c)))
         (setf (action-result-success action) nil)
         (setf (action-result action) (format nil "Error ~/print-error-message/ occurred during execute-command for command ~s and parameters ~s" x (dispatch-command-name c) parameters))
         (setf (action-complete action) t)))
    (progn
         (bt:condition-notify (action-cv action))
         (bt:release-lock (action-lock action)))))

(defun determine-current-model (model-or-name)
  (if (act-r-model-p model-or-name)
      model-or-name
    (get-model model-or-name)))

(defun evaluate-command (action command parameters)
      (handler-case
        (if (eq :lisp (dispatch-command-evaluator command))
            (aif (determine-current-model (action-model action))
                 (let ((*current-act-r-model* it))
                   (let ((r (multiple-value-list (apply (dispatch-command-underlying-function command) parameters))))
                     (values-list (cons t r))))
                 (let ((r (multiple-value-list (apply (dispatch-command-underlying-function command) parameters))))
                   (values-list (cons t r))))
          (send-remote-command (dispatch-command-evaluator command) action command parameters))
      (error (x) 
       (values nil (format nil "Error ~/print-error-message/ occurred while trying to evaluate command ~s with parameters ~s" x (dispatch-command-name command) parameters)))))

(defun model-to-name (name)
  (if (stringp name)
      name
    (if (act-r-model-p name)
        (act-r-model-name name)
      nil)))

(defun send-remote-command (handler action command parameters)
  (let (p)
    (bt:with-lock-held ((handler-sent-requests-lock handler))
      (setf p (make-pending-request :action action :id (incf (handler-id handler))))
      (push p (handler-sent-requests handler)))
    (bt:acquire-lock (pending-request-lock p))
    (handler-case 
        (progn
          (bt:with-lock-held ((handler-stream-lock handler))
            (format (usocket:socket-stream (handler-socket handler)) "{\"method\": \"evaluate\", \"params\": ~a, \"id\": ~d}~c" 
              (json:encode-json-to-string (cons (dispatch-command-underlying-function command) (cons (model-to-name (action-model action)) parameters))) (pending-request-id p) (code-char 4))
            (force-output (usocket:socket-stream (handler-socket handler))))
          (loop
            (when (pending-request-complete p)
              (return-from send-remote-command (values-list (append (list (pending-request-success p)) (if (listp (pending-request-result p))
                                                                                                           (pending-request-result p)
                                                                                                         (list (pending-request-result p)))))) ;(values (pending-request-success p) (pending-request-result p)))
              )
            (bt:condition-wait (pending-request-cv p) (pending-request-lock p))))
      (error (x)
        (bt:with-lock-held ((handler-sent-requests-lock handler))
          
          (setf (handler-sent-requests handler) (remove p (handler-sent-requests handler))))
        (bt:release-lock (pending-request-lock p))
        (error (format nil "Error during send-remote-command ~/print-error-message/" x))))))
    

(eval-when (:load-toplevel :execute)
  (cond (*dispatcher* (error "The *dispatcher* variable is already set, but should be nil."))
        (*dont-start-dispatcher* (dont-start-des))
        ((start-des) (format t "~%The ACT-R remote command dispatcher has been started~%"))
        (t (error "~%The ACT-R remote command dispatcher did not start correctly.~%"))))


;;; Add a lock that protects a "class" of things that can't be run
;;; at the same time.  Clear-all feels like it should be in that class,
;;; but there's a problem with doing that since it gets called recursively
;;; when loading/reloading and a recursive lock doesn't provide a 'no-wait'
;;; interface.  So for now, clear-all isn't protected but also isn't made
;;; available remotely.



(defvar *top-level-lock* (bt:make-lock "top-level"))
(defvar *top-level-user* nil)

#+:single-threaded-act-r
(defmacro with-top-level-lock (warning user &body body)
  (declare (ignore warning user))
  `(progn
     ,@body))


#-:single-threaded-act-r
(defmacro with-top-level-lock (warning user &body body)
  `(if (bt:acquire-lock *top-level-lock* nil)
       (unwind-protect 
           (progn
             (setf *top-level-user* ,user)
             ,@body)
         (bt:release-lock *top-level-lock*))
     (print-warning "ACT-R system unavailable because of ~a. ~a" *top-level-user* ,warning)))


;;; Add the commands for the output traces and corresponding commands
;;; since they can't go into misc-utils where they're defined because
;;; that's loaded before this.

(defun output-command-dummy (string)
  (declare (ignorable string))
  nil)

(add-act-r-command "warning-trace" nil "Output of the ACT-R warning commands for monitoring - shouldn't be evaluated. Params: warning-string.")

(add-act-r-command "model-trace" nil "Output of the ACT-R model for monitoring - shouldn't be evaluated. Params: model-output-string.")

(add-act-r-command "command-trace" nil "Output of the ACT-R commands for monitoring - shouldn't be evaluated. Params: command-output-string.")

(add-act-r-command "general-trace" nil "Output from the act-r-output command for monitoring - shouldn't be evaluated. Params: command-output-string.")

(defun print-warning-internal (string)
  (when string
    (unless (bt:with-recursive-lock-held ((printing-module-lock *act-r-warning-capture*))
              (if (printing-module-capture-warnings *act-r-warning-capture*)
                  (push-last string (printing-module-captured-warnings *act-r-warning-capture*))
                nil))
      (evaluate-act-r-command "warning-trace" (format nil "#|Warning~:[~*~;~@[ (in model ~a)~]~]: ~a |#~%" (> (length (mp-models)) 1) (current-model) string)))
    nil))

(add-act-r-command "print-warning" 'print-warning-internal "Send a string to the ACT-R warning-trace. Params: warning-string." nil)
(add-act-r-command "model-output" 'model-output-internal "Send a string to the ACT-R model-trace. Params: model-output-string." nil)
(add-act-r-command "command-output" 'command-output-internal "Send a string to the ACT-R command-trace. Params: command-output-string." nil)
(add-act-r-command "model-warning" 'model-warning-internal "Send a string to the ACT-R warning-trace. Params: warning-string." nil)
(add-act-r-command "one-time-model-warning" 'one-time-model-warning-internal "Send a string to the ACT-R warning-trace only once per tag. Params: tag warning-string.")
(add-act-r-command "act-r-output" 'act-r-output-internal "Send a string to the ACT-R general-trace. Params: output-string." nil)


;;; A command for receiving a unique name that can be used for a command

(defun get-new-command-name (&optional base)
  (bt:with-lock-held ((dispatcher-command-lock *dispatcher*))
    
    (do* ((b (cond ((stringp base) base)
                   ((symbolp base) (symbol-name base))
                   (t "command")))
          (name (symbol-name (gensym b)) (symbol-name (gensym b))))
         ((not (and (gethash name (dispatcher-command-table *dispatcher*))
                    (gethash name (dispatcher-spec-names *dispatcher*))))
          (progn 
            (setf (gethash name (dispatcher-spec-names *dispatcher*)) t)
            name)))))

(add-act-r-command "get-new-command-name" 'get-new-command-name "Get a unique name that can be used to create a command. Params: {name-stem}." t)


;;; 
;;; Send all the ACT-R output to the current *standard-output*.
;;;
;;; Really should have a lock around these, but that's a heavy price
;;; and Lisp users are more likely to use the ACT-R controls, like
;;; setting :v, :cmdt, and no-output.


(defvar *act-r-echo-stream* nil)
(defvar *act-r-echo-command* nil)

;;; This is now simple...
(defun echo-trace-stream (output)
  (format *act-r-echo-stream* "~a" output))

#|

Some options for CCL GUI output control other than AltConsole.
Going to use the initial listener approach by default

Goes to initial listener only:

(setf *act-r-echo-stream* (GUI::cocoa-listener-process-output-stream (GUI::top-listener-process))) 

Goes to 'active' listener:

#+(and :ccl :darwin :hemlock) (defun echo-trace-stream (output)
                                (when *act-r-echo-stream*
                                  (HI::call-with-output-to-listener 
                                   (lambda () (format t "~a" output)))))

|#


(defun echo-act-r-output ()
  (awhen *act-r-echo-command*
         (remove-act-r-command-monitor "model-trace" it)
         (remove-act-r-command-monitor "command-trace" it)
         (remove-act-r-command-monitor "warning-trace" it)
         (remove-act-r-command-monitor "general-trace" it))

  (let ((name (get-new-command-name "echo-output")))
    (add-act-r-command name 'echo-trace-stream "Internal command for monitoring output - should not be evaluated." nil)

    ;; special hack for CCL GUI to force output to initial listener
    ;; instead of the AltConsole window
    #+(and :ccl :darwin :hemlock)
    (setf *act-r-echo-stream* (GUI::cocoa-listener-process-output-stream (GUI::top-listener-process))) 

    #-(and :ccl :darwin :hemlock)
    (setf *act-r-echo-stream* *standard-output*)

    (monitor-act-r-command "model-trace" name)
    (monitor-act-r-command "command-trace" name)
    (monitor-act-r-command "warning-trace" name)
    (monitor-act-r-command "general-trace" name)
    (setf *act-r-echo-command* name)))
    

(defmacro suppress-act-r-output (&body body)
  "Just suppress command output while evaluating ACT-R commands but don't remove the monitors"
  (let ((current (gensym)))
    `(let ((,current *act-r-echo-stream*))
       (setf *act-r-echo-stream* nil) ;; don't just do it in the let because there could be threading issues
       (unwind-protect 
           (progn ,@body)
         (setf *act-r-echo-stream* ,current)))))

(defun turn-off-act-r-output ()
  (awhen *act-r-echo-command*
         (remove-act-r-command-monitor "model-trace" it)
         (remove-act-r-command-monitor "command-trace" it)
         (remove-act-r-command-monitor "warning-trace" it)
         (remove-act-r-command-monitor "general-trace" it)
         (remove-act-r-command it)
         (setf *act-r-echo-command* nil)
         (setf *act-r-echo-stream* nil)
         t))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Add a remote command for loading a model file.
;;;

;;; Need a way to determine if the load is coming through the dispatcher
;;; or not because how errors are handled matter -- can't invoke the
;;; debugger from a background thread safely so need to abort in that
;;; situation from code that's trying to handle errors (which is only
;;; define-model at this point).

(defvar *top-level-load* t)

(defun top-level-load? ()
  *top-level-load*)


(defun internal-load-act-r-model (file &optional compile)
  "Loads the file specified after translating the pathname and captures all *standard-output* and *error-output* sending it to the ACT-R warning trace"
  (let ((path (translate-logical-pathname file)))
    (if (not (probe-file path))
        (error (format nil "File ~s which translates to ~s does not exist." file path))
      (with-top-level-lock "Cannot load a model file." "load-act-r-model"
        (let* ((save-stream (make-string-output-stream))
               (display-stream (make-broadcast-stream *standard-output* save-stream))
               (error-stream (make-broadcast-stream *error-output* save-stream))
               (*standard-output* display-stream)
               (*error-output* error-stream)
               (*top-level-load* nil))              
          (handler-case
              (if compile
                  (compile-and-load path)
                (load path))
            (error (x)
              (error (format nil "Error ~/print-error-message/ while trying to load file ~s" x file))))
          (let ((s (get-output-stream-string save-stream)))
            (unless (zerop (length s))
              (print-warning "Non-ACT-R messages during load of ~s:~%~a~%" file s))
            t))))))

;; and a separate one for general code because with the split of task and model
;; the tutorial units use load-act-r-model to load the model file which means
;; that file couldn't be loaded with load-act-r-model, and the Environment
;; needs a way to 'load' Lisp files which contain a task and a call to load-act-r-model.


(defun internal-load-act-r-code (file &optional compile)
  "Loads the file specified after translating the pathname and captures all *standard-output* and *error-output* sending it to the ACT-R warning trace"
  (let ((path (translate-logical-pathname file)))
    (if (not (probe-file path))
        (error (format nil "File ~s which translates to ~s does not exist." file path))
      (let* ((save-stream (make-string-output-stream))
             (display-stream (make-broadcast-stream *standard-output* save-stream))
             (error-stream (make-broadcast-stream *error-output* save-stream))
             (*standard-output* display-stream)
             (*error-output* error-stream)
             (*top-level-load* nil))              
        (handler-case
            (if compile
                (compile-and-load path)
              (load path))
          (error (x)
            (error (format nil "Error ~/print-error-message/ while trying to load file ~s" x file))))
        (let ((s (get-output-stream-string save-stream)))
          (unless (zerop (length s))
            (print-warning "Non-ACT-R messages during load of ~s:~%~a~%" file s))
          t)))))



(add-act-r-command "load-act-r-model" 'internal-load-act-r-model "Loads the ACT-R model file indicated. Params: pathname {compile}." "Load-act-r-model currently loading a model")
(add-act-r-command "load-act-r-code" 'internal-load-act-r-code "Loads the Lisp code file indicated. Params: pathname {compile}." "Load-act-r-code currently loading a file")

(defun load-act-r-model (file &optional compile)
  (dispatch-apply "load-act-r-model" file compile))

(defun load-act-r-code (file &optional compile)
  (dispatch-apply "load-act-r-code" file compile))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Code to parse strings into symbols except a string which starts and ends with
;;; single quotes.  Those strings will be converted to a string without the single
;;; quotes.  For the symbols, the string is upcased and then interned into the
;;; value of *default-package* unless the first character is a colon.  When it
;;; starts with a colon the colon is removed and it's interned in the keyword
;;; package. 
;;; Value passed can be a single item or a (possibly) nested list of items.
;;; A new list of items is returned if it is a list.
;;; Numbers, t, and nil are valid items and will be unchanged.


(defun decode-string-names (sn)
  (cond ((null sn)
         nil)
        ((or (symbolp sn) (numberp sn))
         sn)
        ((stringp sn)
         (decode-string sn))
        ((listp sn)
         (mapcar 'decode-string-names sn))
        (t
         (error "Invalid item ~s found when parsing strings into ACT-R elements." sn))))


;;; Also include a recursive string->name that doesn't decode the string values

(defun string->name-recursive (s)
  (cond ((listp s)
         (mapcar 'string->name-recursive s))
        (t
         (string->name s))))

(defun encode-string-names (sn)
  (cond ((null sn)
         nil)
        ((or (symbolp sn) (numberp sn))
         sn)
        ((stringp sn)
         (format nil "'~a'" sn))
        ((listp sn)
         (mapcar 'encode-string-names sn))
        (t
         (error "Invalid item ~s found when encoding strings." sn))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Anything which needs to eval something that might be a dispatch function or
;;; a real function should use one of these.

(defun dispatch-eval (form)
  (if (listp form)
      (if (stringp (first form))
          (let ((results (multiple-value-list (apply 'evaluate-act-r-command 
                                                     (mapcar (lambda (x) (if (and (listp x) (first x) (fctornil (first x)))
                                                                             (eval x)
                                                                           x))
                                                       form)))))
            (if (first results)
                (values-list (rest results))
              (print-warning "Error ~s while attempting to evaluate the form ~s" (second results) form)))
        (eval form))
    (eval form)))


(defun dispatch-eval-names (form)
  (if (listp form)
      (if (stringp (first form))
          (let ((results (multiple-value-list (apply 'evaluate-act-r-command 
                                                     (mapcar (lambda (x) (if (and (listp x) (first x) (fctornil (first x)))
                                                                             (eval x)
                                                                           x))
                                                       form)))))
            (if (first results)
                (values-list (mapcar 'decode-string (rest results)))
              (print-warning "Error ~s while attempting to evaluate the form ~s" (second results) form)))
        (eval form))
    (eval form)))


(defun dispatch-apply (fct &rest rest)
  (cond ((stringp fct)
         (let ((results (multiple-value-list (apply 'evaluate-act-r-command (cons fct rest)))))
           (if (first results)
               (values-list (rest results))
             (print-warning "Error ~s while attempting to evaluate the form (~s ~{~s~^ ~})" (second results) fct rest))))
        ((functionp fct)
         (apply fct rest))
        ((and (symbolp fct) (fboundp fct))
         (apply fct rest))
        (t
         (print-warning "Function ~s passed to dispatch-apply is not a local function or valid remote command string" fct))))

(defun dispatch-apply-names (fct &rest rest)
  (cond ((stringp fct)
         (let ((results (multiple-value-list (apply 'evaluate-act-r-command (cons fct rest)))))
           (if (first results)
               (values-list (mapcar 'decode-string (rest results)))
             (print-warning "Error ~s while attempting to evaluate the form (~s ~{~s~^ ~})" (second results) fct rest))))
        ((functionp fct)
         (apply fct rest))
        ((and (symbolp fct) (fboundp fct))
         (apply fct rest))
        (t
         (print-warning "Function ~s provide for dispatch-apply is not a local or dispatcher function" fct))))

(defun dispatch-apply-list (fct param-list)
  (cond ((and (symbolp fct) (fboundp fct))
         (apply fct param-list))
        ((functionp fct)
         (apply fct param-list))
        ((stringp fct)
         (let ((results (multiple-value-list (apply 'evaluate-act-r-command (cons fct param-list)))))
           (if (first results)
               (values-list (rest results))
             (print-warning "Error ~s while attempting to evaluate the form (~s ~s})" (second results) fct param-list))))
        
        (t
         (print-warning "Function ~s provide for dispatch-apply-list is not a local or dispatcher function" fct))))

(defun dispatch-apply-list-names (fct param-list)
  (cond ((stringp fct)
         (let ((results (multiple-value-list (apply 'evaluate-act-r-command (cons fct param-list)))))
           (if (first results)
               (values-list (mapcar 'decode-string (rest results)))
             (print-warning "Error ~s while attempting to evaluate the form (~s ~s})" (second results) fct param-list))))
        ((functionp fct)
         (apply fct param-list))
        ((and (symbolp fct) (fboundp fct))
         (apply fct param-list))
        (t
         (print-warning "Function ~s provide for dispatch-apply-list is not a local or dispatcher function" fct))))

;;;; Process an options-list into a flat list of keyword value pairs

(defun process-options-list (ol cmd valid)
  (cond ((null ol)
         (values t nil))
        ((and (every (lambda (x) (and (consp x) (keywordp (car x)))) ol)
              (= (length ol) (length (remove-duplicates ol :key 'car))))
         (if (every (lambda (x) (find (car x) valid)) ol)
             (values t (mapcan (lambda (x) (list (car x) (cdr x))) ol))
           (print-warning "Invalid option name in options list ~s for command ~s.  Valid values are ~s." ol cmd valid)))
        ((and (every (lambda (x) (and (listp x) (= (length x) 2) (stringp (first x)))) ol)
              (= (length ol) (length (remove-duplicates ol :key 'first :test 'string-equal))))
         
         (if (let ((v (mapcar 'symbol-name valid)))
               (every (lambda (x) (find (first x) v :test 'string-equal)) ol))
             (values t (mapcan (lambda (x) (list (read-from-string (format nil ":~a" (first x))) (second x))) ol))
           (print-warning "Invalid option name in options list ~s for command ~s.  Valid values are ~s." ol cmd valid)))
        (t
         (print-warning "Invalid options-list ~s passed to command ~a." ol cmd))))

(defun convert-options-list-items (ol s->n d-s)
  (when ol
    (do* ((items ol (cdr items))
          (item (car items) (car items))
          (name t (not name))
          (convert (if (find item s->n)
                       :string
                     (if (find item d-s)
                         :decode
                       nil))
                   (if name
                       (if (find item s->n)
                           :string
                         (if (find item d-s)
                             :decode
                           nil))
                     convert))
          (res (list item) (push-last (if (and (not name) convert)
                                          (if (eq convert :string)
                                              (string->name-recursive item)
                                            (decode-string-names item))
                                        item) res)))
         ((null (cdr items)) res))))



;;;;;; Here's a test which can be used to detect real functions or valid dispatch functions

(defun local-or-remote-function-p (function?)
  (and
   (or (functionp function?) 
       (and (symbolp function?) (fboundp function?))
       (and (stringp function?) (check-act-r-command function?)))
   t))


(defun local-or-remote-function-or-nil (function?)
  (and 
   (or (null function?)
       (functionp function?) 
       (and (symbolp function?) (fboundp function?))
       (and (stringp function?) (check-act-r-command function?)))
   t))
    
;;;;;; This is the 'user' command which should be used if calling from a model defintion

(defun call-act-r-command (command &rest rest)
  (multiple-value-bind (success result) (apply 'evaluate-act-r-command (cons command rest))
    (if success
        result
      (print-warning "Error ~s while attempting to call the ACT-R command (~s ~{~s~^ ~})" result command rest))))


;;;;;;; This needs to be here for running the human version of the tutorial experiments ;;;;;;;

(defun process-events ()
  (bt:thread-yield))


;;;; Add a command for getting the ACT-R version string

(defun act-r-version-string ()
  *actr-version-string*)

(add-act-r-command "act-r-version" 'act-r-version-string "Returns the current ACT-R version string. No params." nil)


;;; Actually echo the output by default now if it's not the standalone.

#-:standalone (echo-act-r-output)

;;; The single-threaded code currently sets :standalone mode
;;; to avoid starting the dispatcher so need to handle that too.

#+:single-threaded-act-r (echo-act-r-output)


#|
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
|#

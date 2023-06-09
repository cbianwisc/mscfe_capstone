VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ImplVolOptPrice"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Option Explicit

Dim genId1 As Long
Dim genId2 As Long

Dim contractColumnsArray() As Variant

' constants
Const STR_SHEET_NAME = "ImplVolOptPrice"
Const STR_CALCULATE_IMPLIED_VOLATILITY = "calculateImpliedVolatility"
Const STR_CANCEL_CALCULATE_IMPLIED_VOLATILITY = "cancelCalculateImpliedVolatility"
Const STR_CALCULATE_IMPLIED_VOLATILITY_TICK = "calculateImpliedVolatilityTick"
Const STR_IMPLIED_VOLATILITY = "custOptCompImpliedVol"
Const STR_CALCULATE_OPTION_PRICE = "calculateOptionPrice"
Const STR_CANCEL_CALCULATE_OPTION_PRICE = "cancelCalculateOptionPrice"
Const STR_CALCULATE_OPTION_PRICE_TICK = "calculateOptionPriceTick"
Const STR_OPTION_PRICE = "custOptCompOptPrice"
Const STR_TICK_ATTRIB = "custOptCompTickAttrib"


' cells
Const CELL_SERVER_NAME = "B5" ' cell with server name

' columns
Const startOfContractColumns = 1 ' contract first column index (symbol)
Const calculateImpliedVolatilityOptionPriceColumnIndex = 13 ' index of "option price" column (calculateImpliedVolatility)
Const calculateImpliedVolatilityUnderPriceColumnIndex = 14 ' index of "underlying price" column (calculateImpliedVolatility)
Const calculateImpliedVolatilityStatusColumnIndex = 15 ' index of "status" column (calculateImpliedVolatility)
Const calculateImpliedVolatilityIdColumnIndex = 16 ' index of "id" column (calculateImpliedVolatility)
Const calculateImpliedVolatilityErrorColumnIndex = 17 ' index of "error" column (calculateImpliedVolatility)
Const calculateImpliedVolatilityResultTickAttrib = 18 ' index of "tick attrib" column (calculateImpliedVolatility)
Const calculateImpliedVolatilityResultColumnIndex = 19 ' index of "implied volatility" column (calculateImpliedVolatility)

Const calculateOptionPriceImpliedVolatilityColumnIndex = 20 ' index of "implied volatility" column (calculateOptionPrice)
Const calculateOptionPriceUnderPriceColumnIndex = 21 ' index of "underlying price" column (calculateOptionPrice)
Const calculateOptionPriceStatusColumnIndex = 22 ' index of "status" column (calculateOptionPrice)
Const calculateOptionPriceIdColumnIndex = 23 ' index of "id" column (calculateOptionPrice)
Const calculateOptionPriceErrorColumnIndex = 24 ' index of "error" column (calculateOptionPrice)
Const calculateOptionPriceResultTickAttrib = 25 ' index of "tick attrib" column (calculateOptionPrice)
Const calculateOptionPriceResultColumnIndex = 26 ' index of "option price" column (calculateOptionPrice)

' rows
Const dataStartRowIndex = 10 ' starting row of data
Const dataEndRowIndex = util.maxRowsToFormat ' ending row of data

' ========================================================
' contract columns
' ========================================================
Private Function getContractColumns() As Variant()
    If (Not Not contractColumnsArray) <> 0 Then
        ' do not re-initialize array
        GoTo getContractColumnsEnd
    End If

    contractColumnsArray = Array("SYMBOL", "SECTYPE", "LASTTRADEDATE", "STRIKE", "RIGHT", "MULTIPLIER", "TRADINGCLASS", "EXCH", "PRIMEXCH", "CURRENCY", "LOCALSYMBOL", "CONID")

getContractColumnsEnd:
    getContractColumns = contractColumnsArray
End Function

' ========================================================
' cancel calculate implied volatility for active row when button is pressed
' ========================================================
Sub cancelCalculateImpliedVolatility()
    cancelCalculateRequest STR_CANCEL_CALCULATE_IMPLIED_VOLATILITY
End Sub

' ========================================================
' cancel calculate option price for active row when button is pressed
' ========================================================
Sub cancelCalculateOptionPrice()
    cancelCalculateRequest STR_CANCEL_CALCULATE_OPTION_PRICE
End Sub

' ========================================================
' send cancel calculate request
' ========================================================
Sub cancelCalculateRequest(requestType As String)
    
    Dim idColumnIndex As Integer, statusColumnIndex As Integer, errorColumnIndex As Integer, resultColumnIndex As Integer, tickAttribColumnIndex As Integer

    If requestType = STR_CANCEL_CALCULATE_OPTION_PRICE Then
        idColumnIndex = calculateOptionPriceIdColumnIndex
        statusColumnIndex = calculateOptionPriceStatusColumnIndex
        errorColumnIndex = calculateOptionPriceErrorColumnIndex
        tickAttribColumnIndex = calculateOptionPriceResultTickAttrib
        resultColumnIndex = calculateOptionPriceResultColumnIndex
    Else
        idColumnIndex = calculateImpliedVolatilityIdColumnIndex
        statusColumnIndex = calculateImpliedVolatilityStatusColumnIndex
        errorColumnIndex = calculateImpliedVolatilityErrorColumnIndex
        tickAttribColumnIndex = calculateImpliedVolatilityResultTickAttrib
        resultColumnIndex = calculateImpliedVolatilityResultColumnIndex
    End If

    Dim server As String, lastRowIndex As Integer, row As Range, id As String

    server = util.getServerVal(STR_SHEET_NAME, CELL_SERVER_NAME)
    If server = util.STR_EMPTY Then Exit Sub

    With Worksheets(STR_SHEET_NAME)

        For Each row In Selection.rows
            lastRowIndex = row.row

            If .Cells(row.row, idColumnIndex).value = STR_EMPTY Then GoTo Continue
            If Not util.hasContractData(Worksheets(STR_SHEET_NAME), dataStartRowIndex, row, startOfContractColumns, getContractColumns()) Then GoTo Continue
        
            id = .Cells(row.row, idColumnIndex).value
        
            ' clear columns
            .Cells(row.row, statusColumnIndex).ClearContents
            .Cells(row.row, idColumnIndex).ClearContents
            .Cells(row.row, errorColumnIndex).ClearContents
            .Cells(row.row, tickAttribColumnIndex).ClearContents
            .Cells(row.row, resultColumnIndex).ClearContents
        
            util.sendRequest server, requestType, id
Continue:
        Next row

        .Cells(lastRowIndex, 1).offset(1, 0).Activate
    End With
End Sub


' ========================================================
' calculate implied volatility for active row when button is pressed
' ========================================================
Sub calculateImpliedVolatility()
    Dim server As String, lastRowIndex As Integer, row As Range

    server = util.getServerVal(STR_SHEET_NAME, CELL_SERVER_NAME)
    If server = STR_EMPTY Then Exit Sub

    With Worksheets(STR_SHEET_NAME)
        For Each row In Selection.rows
            lastRowIndex = row.row
    
            If .Cells(row.row, calculateImpliedVolatilityIdColumnIndex).value <> STR_EMPTY Then GoTo Continue
            If Not util.hasContractData(Worksheets(STR_SHEET_NAME), dataStartRowIndex, row, startOfContractColumns, getContractColumns()) Then GoTo Continue
            sendCalculateRequest server, row, STR_CALCULATE_IMPLIED_VOLATILITY
            .Cells(lastRowIndex, 1).offset(1, 0).Activate
Continue:
        Next row
    End With

End Sub

' ========================================================
' calculate option price for active row when button is pressed
' ========================================================
Sub calculateOptionPrice()
    Dim server As String, lastRowIndex As Integer, row As Range

    server = util.getServerVal(STR_SHEET_NAME, CELL_SERVER_NAME)
    If server = STR_EMPTY Then Exit Sub

    With Worksheets(STR_SHEET_NAME)
        For Each row In Selection.rows
            lastRowIndex = row.row
            If .Cells(row.row, calculateOptionPriceIdColumnIndex).value <> STR_EMPTY Then GoTo Continue
            If Not util.hasContractData(Worksheets(STR_SHEET_NAME), dataStartRowIndex, row, startOfContractColumns, getContractColumns()) Then GoTo Continue
            sendCalculateRequest server, row, STR_CALCULATE_OPTION_PRICE
            .Cells(lastRowIndex, 1).offset(1, 0).Activate
Continue:
        Next row
    End With

End Sub

' ========================================================
' send calculate option price
' ========================================================
Sub sendCalculateRequest(server As String, cell As Range, requestType As String)

    Dim idColumnIndex As Integer, statusColumnIndex As Integer, errorColumnIndex As Integer, resultColumnIndex As Integer
    Dim firstValColumnIndex As Integer, secondValColumnIndex As Integer, tickAttribColumnIndex As Integer
    Dim tickStr As String, resultStr As String, tickAttrib As String
    Dim id As String
    
    If requestType = STR_CALCULATE_OPTION_PRICE Then
        idColumnIndex = calculateOptionPriceIdColumnIndex
        statusColumnIndex = calculateOptionPriceStatusColumnIndex
        errorColumnIndex = calculateOptionPriceErrorColumnIndex
        tickAttribColumnIndex = calculateOptionPriceResultTickAttrib
        resultColumnIndex = calculateOptionPriceResultColumnIndex
        firstValColumnIndex = calculateOptionPriceImpliedVolatilityColumnIndex
        secondValColumnIndex = calculateOptionPriceUnderPriceColumnIndex
        tickStr = STR_CALCULATE_OPTION_PRICE_TICK
        tickAttrib = STR_TICK_ATTRIB
        resultStr = STR_OPTION_PRICE
        id = util.getIDpost(genId1, util.ID_CALCULATE_OPTION_PRICE)
    Else
        idColumnIndex = calculateImpliedVolatilityIdColumnIndex
        statusColumnIndex = calculateImpliedVolatilityStatusColumnIndex
        errorColumnIndex = calculateImpliedVolatilityErrorColumnIndex
        tickAttribColumnIndex = calculateImpliedVolatilityResultTickAttrib
        resultColumnIndex = calculateImpliedVolatilityResultColumnIndex
        firstValColumnIndex = calculateImpliedVolatilityOptionPriceColumnIndex
        secondValColumnIndex = calculateImpliedVolatilityUnderPriceColumnIndex
        tickStr = STR_CALCULATE_IMPLIED_VOLATILITY_TICK
        tickAttrib = STR_TICK_ATTRIB
        resultStr = STR_IMPLIED_VOLATILITY
        id = util.getIDpost(genId2, util.ID_CALCULATE_IMPLIED_VOLATILITY)
    End If

    Dim firstVal As String
    Dim secondVal As String

    With Worksheets(STR_SHEET_NAME)
        .Cells(cell.row, idColumnIndex).value = id

        firstVal = .Cells(cell.row, firstValColumnIndex).value
        secondVal = .Cells(cell.row, secondValColumnIndex).value

        ' fill status column with formula
        .Cells(cell.row, statusColumnIndex).Formula = util.composeLink(server, tickStr, id, util.STR_STATUS)
        If util.cleanOnError(.Cells(cell.row, statusColumnIndex)) Then
            .Cells(cell.row, statusColumnIndex).ClearContents
            .Cells(cell.row, idColumnIndex).ClearContents
            .Cells(cell.row, errorColumnIndex).ClearContents
            .Cells(cell.row, tickAttribColumnIndex).ClearContents
            .Cells(cell.row, resultColumnIndex).ClearContents
            Exit Sub
        End If

        ' send request
        util.sendPoke Worksheets(STR_SHEET_NAME), server, requestType, id & util.QMARK & firstVal & util.QMARK & secondVal, cell, startOfContractColumns, getContractColumns(), 0, calculateOptionPriceIdColumnIndex, 0, 0, 0, 0

        ' fill error column with formula
        .Cells(cell.row, errorColumnIndex).Formula = util.composeLink(server, tickStr, id, util.STR_ERROR)
        ' fill tick attrib column with formula
        .Cells(cell.row, tickAttribColumnIndex).Formula = util.composeLink(server, tickStr, id, tickAttrib)
        ' fill result column with formula
        .Cells(cell.row, resultColumnIndex).Formula = util.composeLink(server, tickStr, id, resultStr)

    End With

End Sub

' ========================================================
' cancel all calculate requests for all rows (called when workbook is closed)
' ========================================================
Sub cancelAllCalculateRequestsOnExit()
    cancelAllCalculateRequests STR_CANCEL_CALCULATE_IMPLIED_VOLATILITY
    cancelAllCalculateRequests STR_CANCEL_CALCULATE_OPTION_PRICE
End Sub

' ========================================================
' cancel all calculate requests for all rows
' ========================================================
Sub cancelAllCalculateRequests(requestType As String)

    Dim idColumnIndex As Integer
    
    If requestType = STR_CANCEL_CALCULATE_OPTION_PRICE Then
        idColumnIndex = calculateOptionPriceIdColumnIndex
    Else
        idColumnIndex = calculateImpliedVolatilityIdColumnIndex
    End If

    Dim rng As Range, row As Range, cell As Range
    Dim server As String
    Dim id As String

    server = util.getServerVal(STR_SHEET_NAME, CELL_SERVER_NAME)
    If server = util.STR_EMPTY Then Exit Sub

    With Worksheets(STR_SHEET_NAME)
        Set rng = .Range(.Cells(dataStartRowIndex, idColumnIndex), .Cells(dataEndRowIndex, idColumnIndex))

        For Each row In rng.rows
            For Each cell In row.Cells
            If InStr(cell.value, util.IDENTIFIER_PREFIX) <> 0 Then
                util.sendRequest server, requestType, cell.value
            End If
          Next cell
        Next row
    End With
End Sub

' ========================================================
' re-request all calculate requests for all rows (called when workbook is closed)
' ========================================================
Sub requestAllCalculateRequestsOnStart()
    requestAllCalculateRequests STR_CALCULATE_IMPLIED_VOLATILITY
    requestAllCalculateRequests STR_CALCULATE_OPTION_PRICE
End Sub

' ========================================================
' re-request all calculate requests for all rows
' ========================================================
Sub requestAllCalculateRequests(requestType As String)

    Dim idColumnIndex As Integer
    
    If requestType = STR_CALCULATE_OPTION_PRICE Then
        idColumnIndex = calculateOptionPriceIdColumnIndex
    Else
        idColumnIndex = calculateImpliedVolatilityIdColumnIndex
    End If

    Dim server As String
    server = util.getServerVal(STR_SHEET_NAME, CELL_SERVER_NAME)
    If server = util.STR_EMPTY Then Exit Sub

    Dim rng As Range

    With Worksheets(STR_SHEET_NAME)
        Set rng = .Range(.Cells(dataStartRowIndex, idColumnIndex), .Cells(dataEndRowIndex, idColumnIndex))

        Dim row As Range, cell As Range
        For Each row In rng.rows
            For Each cell In row.Cells
            If InStr(cell.value, util.IDENTIFIER_PREFIX) <> 0 Then
                sendCalculateRequest server, cell, requestType
            End If
          Next cell
        Next row
    End With
End Sub


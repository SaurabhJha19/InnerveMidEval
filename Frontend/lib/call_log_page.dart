import 'package:flutter/material.dart';
import 'package:call_log/call_log.dart';
import 'package:permission_handler/permission_handler.dart';

class CallLogScreen extends StatefulWidget {
  @override
  _CallLogScreenState createState() => _CallLogScreenState();
}

class _CallLogScreenState extends State<CallLogScreen> {
  List<CallLogEntry> callLogs = []; // Stores all call logs
  List<CallLogEntry> filteredLogs = []; // Stores filtered call logs
  TextEditingController _searchController = TextEditingController(); // Controller for search

  @override
  void initState() {
    super.initState();
    _fetchCallLogs();
    _searchController.addListener(_filterCallLogs);
  }

  // Fetch call logs from the device
  Future<void> _fetchCallLogs() async {
    var status = await Permission.phone.request(); // Request permission

    if (status.isGranted) {
      Iterable<CallLogEntry> logs = await CallLog.get();
      setState(() {
        callLogs = logs.toList();
        filteredLogs = callLogs; // Initially, show all logs
      });
    } else {
      print("Permission denied");
    }
  }

  // Filter call logs based on search input
  void _filterCallLogs() {
    String query = _searchController.text.toLowerCase();
    setState(() {
      filteredLogs = callLogs.where((call) {
        String name = call.name?.toLowerCase() ?? "";
        String number = call.number?.toLowerCase() ?? "";
        return name.contains(query) || number.contains(query);
      }).toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Call Log")),
      body: Column(
        children: [
          // Search Bar
          Padding(
            padding: EdgeInsets.all(8.0),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: "Search by name or number...",
                prefixIcon: Icon(Icons.search),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
              ),
            ),
          ),

          // Call Log List
          Expanded(
            child: filteredLogs.isEmpty
                ? Center(child: Text("No calls found"))
                : ListView.builder(
                    itemCount: filteredLogs.length,
                    itemBuilder: (context, index) {
                      CallLogEntry call = filteredLogs[index];
                      return ListTile(
                        leading: Icon(
                          call.callType == CallType.incoming
                              ? Icons.call_received
                              : Icons.call_made,
                          color: call.callType == CallType.missed ? Colors.red : Colors.green,
                        ),
                        title: Text(call.name ?? call.number ?? "Unknown"),
                        subtitle: Text(
                          "${call.callType.toString().split('.').last} - ${DateTime.fromMillisecondsSinceEpoch(call.timestamp ?? 0)}",
                        ),
                        trailing: Text("${call.duration} sec"),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

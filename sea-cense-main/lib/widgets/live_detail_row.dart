// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'package:flutter/material.dart';

class LiveDetailRow extends StatelessWidget {
  final String title;
  final String desc;
  const LiveDetailRow({
    Key? key,
    required this.title,
    required this.desc,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 25),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
                color: Colors.white30, fontSize: 16, letterSpacing: 3.0, fontWeight: FontWeight.bold),
          ),
          const SizedBox(
            height: 5,
          ),
          Text(
            desc,
            style: const TextStyle(
                color: Colors.white30, fontSize: 14, letterSpacing: 3.0, fontWeight: FontWeight.w300),
          ),
        ],
      ),
    );
  }
}

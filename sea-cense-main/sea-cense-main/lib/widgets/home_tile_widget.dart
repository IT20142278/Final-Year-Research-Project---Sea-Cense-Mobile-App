// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'package:flutter/material.dart';
import 'package:sea_cense/style.dart';

class HomeTileWidget extends StatelessWidget {
  final IconData icon;
  final String title;
  final Function()? onTap;
  const HomeTileWidget({
    Key? key,
    required this.icon,
    required this.title,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        width: MediaQuery.of(context).size.width / 3.0,
        height: MediaQuery.of(context).size.width / 3.0,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [
              Color(0xFFF0EFEF),
              Color(0xFFEEE9E9),
            ],
            begin: FractionalOffset(0.0, 0.0),
            end: FractionalOffset(1.0, 0.0),
            stops: [0.0, 1.0],
            tileMode: TileMode.clamp,
          ),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFFC7C7C7).withOpacity(0.5),
              spreadRadius: 2,
              blurRadius: 5,
              offset: const Offset(0, 2), // changes position of shadow
            ),
          ],
          borderRadius: const BorderRadius.all(Radius.circular(10)),
        ),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(
            icon,
            size: 60,
            color: defaultColor,
            shadows: const <Shadow>[Shadow(color: Color(0xFFFFFFFF), blurRadius: 25.0)],
          ),
          Text(
            title,
            style: const TextStyle(color: Colors.black38, fontWeight: FontWeight.w400, fontSize: 18),
          )
        ]),
      ),
    );
  }
}
